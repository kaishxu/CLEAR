import os
import json
import argparse
import numpy as np
from tqdm import tqdm

def to_memmap(args):  #创建memmap磁盘缓存(没有padding)
    collection_size = sum(1 for _ in open(args.tokenized_collection))
    max_seq_length = 512  #仅根据BERT的最大输入长度进行限制
    token_ids = np.memmap(f"{args.output_dir}/token_ids.memmap", dtype='int32', 
        mode='w+', shape=(collection_size, max_seq_length))
    pids = np.memmap(f"{args.output_dir}/pids.memmap", dtype='int32', 
        mode='w+', shape=(collection_size,))
    lengths = np.memmap(f"{args.output_dir}/lengths.memmap", dtype='int32', 
        mode='w+', shape=(collection_size,))

    for idx, line in enumerate(tqdm(open(args.tokenized_collection), 
            desc="collection", total=collection_size)):
        data = json.loads(line)
        assert int(data['id']) == idx
        pids[idx] = idx
        lengths[idx] = len(data['ids'])
        ids = data['ids'][:max_seq_length]  #截取512
        token_ids[idx, :lengths[idx]] = ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized_collection", type=str, 
        default="./data/tokenize/collection.tokenize.json")
    parser.add_argument("--output_dir", type=str, default="./data/collection_memmap")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    to_memmap(args)