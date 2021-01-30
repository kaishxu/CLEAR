import os
import math
import json
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from transformers import BertTokenizer, BertConfig
from torch.utils.data import DataLoader, Dataset
from dataset import Collection, pack_tensor
from modeling import CLEAR_Embedding


logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)


def create_embed_memmap(ids, memmap_dir, dim):
    if not os.path.exists(memmap_dir):
        os.makedirs(memmap_dir)
    embedding_path = f"{memmap_dir}/embedding.memmap"
    id_path = f"{memmap_dir}/ids.memmap"
    embed_open_mode = "r+" if os.path.exists(embedding_path) else "w+"
    id_open_mode = "r+" if  os.path.exists(id_path) else "w+"
    logger.warning(f"Open Mode: embedding-{embed_open_mode} ids-{id_open_mode}")

    embedding_memmap = np.memmap(embedding_path, dtype='float32', 
        mode=embed_open_mode, shape=(len(ids), dim))
    id_memmap = np.memmap(id_path, dtype='int32', 
        mode=id_open_mode, shape=(len(ids),))
    id_memmap[:] = ids
    # not writable
    id_memmap = np.memmap(id_path, dtype='int32', 
        shape=(len(ids),))
    return embedding_memmap, id_memmap


class CLEAR_PreDataset(Dataset):
    def __init__(self,args):
        self.mode = args.mode
        self.args = args
        self.sep_id = args.sep_id

        if args.mode == 'query':
            self.max_length = args.max_query_length
            self.cls_id = args.qry_id
            self.text_tokens = self.load_queries()
            self.ids = list(self.text_tokens.keys())
        elif args.mode == 'doc':
            self.max_length = args.max_doc_length
            self.cls_id = args.doc_id
            self.text_tokens = Collection(args.collection_memmap_dir)
            self.ids = self.text_tokens.pids

    def __len__(self):  
        return len(self.ids)

    def __getitem__(self, item):
        id_ = self.ids[item]
        input_ids = self.text_tokens[id_]
        input_ids = input_ids[:self.max_length]
        input_ids = [self.cls_id] + input_ids + [self.sep_id]
        sample = {
            "input_ids": input_ids,
            "id" : id_
        }
        return sample

    def load_queries(self):
        queries = dict()
        for line in open(f"{self.args.tokenize_dir}/queries.dev.small.json", 'r'):
            data = json.loads(line)
            queries[int(data['id'])] = data['ids']
        return queries


def collate_function(batch):
    input_ids_lst = [x["input_ids"] for x in batch]
    mask_lst = [[1]*len(input_ids) for input_ids in input_ids_lst]
    data = {
        "input_ids": pack_tensor(input_ids_lst, default=0, dtype=torch.int64),
        "mask": pack_tensor(mask_lst, default=0, dtype=torch.int64),
    }
    id_lst = [x['id'] for x in batch]
    return data, id_lst


def generate_embeddings(args, model):    
    if args.mode == "query":
        memmap_dir = args.query_embedding_dir
    else:
        memmap_dir = args.doc_embedding_dir
    dataset = CLEAR_PreDataset(args)

    embedding_memmap, id_memmap = create_embed_memmap(
        dataset.ids, memmap_dir, model.config.hidden_size)
    id2pos = {identity:i for i, identity in enumerate(id_memmap)}
    
    batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_function)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", batch_size)

    start = timer()
    for batch, ids in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            batch = {k:v.to(args.device) for k, v in batch.items()}
            output = model(**batch)
            sequence_embeddings = output.detach().cpu().numpy()
            poses = [id2pos[identity] for identity in ids]
            embedding_memmap[poses] = sequence_embeddings
    end = timer()
    print(args.mode, "time:", end-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--load_model_path", type=str, required=True)
    parser.add_argument("--bert_path", type=str, default='./bert-base-uncased')
    parser.add_argument("--mode", choices=["query", "doc"], required=True)
    parser.add_argument("--output_dir", type=str, default="./data/precompute")

    parser.add_argument("--msmarco_dir", type=str, default=f"./data")
    parser.add_argument("--collection_memmap_dir", type=str, default="./data/collection_memmap")
    parser.add_argument("--tokenize_dir", type=str, default="./data/tokenize")
    parser.add_argument("--max_query_length", type=int, default=20)
    parser.add_argument("--max_doc_length", type=int, default=256)
    parser.add_argument("--per_gpu_batch_size", default=256, type=int)
    args = parser.parse_args()

    args.doc_embedding_dir = f"{args.output_dir}/doc_embedding"
    args.query_embedding_dir = f"{args.output_dir}/query_embedding"

    logger.info(args)

    # Setup CUDA, GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    # New special tokens: [QRY] and [DOC]
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    tokenizer.add_tokens(["[QRY]", "[DOC]"])
    args.qry_id = tokenizer.encode('[QRY]', add_special_tokens=False)[0]
    args.doc_id = tokenizer.encode('[DOC]', add_special_tokens=False)[0]
    args.sep_id = tokenizer.sep_token_id

    config = BertConfig.from_pretrained(args.load_model_path)
    model = CLEAR_Embedding.from_pretrained(args.load_model_path, config=config)
    model.to(args.device)

    logger.info(args)
    generate_embeddings(args, model)
