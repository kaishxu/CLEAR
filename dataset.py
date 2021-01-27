import os
import gc
import math
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from random import sample as random_sample
from collections import defaultdict
from pyserini import search
from pyserini.index import IndexReader
from torch.utils.data import Dataset


class Collection:
    def __init__(self, collection_memmap_dir):
        self.pids = np.memmap(f"{collection_memmap_dir}/pids.memmap", dtype='int32',)  #memmap磁盘缓存, 无需完整读入全部数据
        self.lengths = np.memmap(f"{collection_memmap_dir}/lengths.memmap", dtype='int32',)
        self.collection_size = len(self.pids)
        self.token_ids = np.memmap(f"{collection_memmap_dir}/token_ids.memmap", 
                dtype='int32', shape=(self.collection_size, 512))
    
    def __len__(self):
        return self.collection_size

    def __getitem__(self, item):
        assert self.pids[item] == item
        return self.token_ids[item, :self.lengths[item]].tolist()


class CLEARDataset(Dataset):
    def __init__(self, mode, args):

        self.mode = mode
        self.args = args
        self.load_collection()  #总passage
        self.load_queries()  #总query
        self.load_samples()  #总样本
        self.qry_id = args.qry_id
        self.doc_id = args.doc_id
        self.sep_id = args.sep_id  #特殊token

        self.max_query_length = args.max_query_length
        self.max_doc_length = args.max_doc_length
        gc.collect()

    def __len__(self):
        assert len(self.qids) == len(self.pos_pids) == len(self.neg_pids)
        return len(self.qids)

    def __getitem__(self, item):
        if self.mode == 'train':
            qid, pos_pid, neg_pid, pos_score, neg_score = self.qids[item], self.pos_pids[item], self.neg_pids[item], self.pos_scores[item], self.neg_scores[item]
            query_input_ids, pos_doc_input_ids, neg_doc_input_ids = self.queries[qid], self.collection[pos_pid], self.collection[neg_pid]  #获取完整tokens
            query_input_ids = query_input_ids[:self.max_query_length]  #限制query长度
            query_input_ids = [self.qry_id] + query_input_ids + [self.sep_id]  #补充特殊token
            pos_doc_input_ids = pos_doc_input_ids[:self.max_doc_length]  #限制doc长度
            pos_doc_input_ids = [self.doc_id] + pos_doc_input_ids + [self.sep_id]  #补充特殊token
            neg_doc_input_ids = neg_doc_input_ids[:self.max_doc_length]  #限制doc长度
            neg_doc_input_ids = [self.doc_id] + neg_doc_input_ids + [self.sep_id]  #补充特殊token

            sample = {
                "query_input_ids": query_input_ids,
                "pos_doc_input_ids": pos_doc_input_ids,
                "neg_doc_input_ids": neg_doc_input_ids,
                "pos_score": pos_score,
                "neg_score": neg_score,
            }  #以字典形式输出
            return sample  #一份样本包含q, pos_d, neg_d
        else:
            qid, pid, score = self.qids[item], self.pids[item], self.scores[item]
            query_input_ids, doc_input_ids = self.queries[qid], self.collection[pid]  #获取完整tokens
            query_input_ids = query_input_ids[:self.max_query_length]  #限制query长度
            query_input_ids = [self.qry_id] + query_input_ids + [self.sep_id]  #补充特殊token
            doc_input_ids = doc_input_ids[:self.max_doc_length]  #限制doc长度
            doc_input_ids = [self.doc_id] + doc_input_ids + [self.sep_id]  #补充特殊token

            sample = {
                "query_input_ids": query_input_ids,
                "doc_input_ids": doc_input_ids,
                "qid": qid,
                "pid" : pid,
                "score": score,
            }  #以字典形式输出
            return sample  #一份样本包含q, d          

    def load_collection(self):
        self.collection = Collection(self.args.collection_memmap_dir)

    def load_queries(self):
        self.queries = dict()
        for line in tqdm(open(f"{self.args.tokenize_dir}/queries.{self.mode}.json", 'r'), desc="queries"):
            data = json.loads(line)
            self.queries[data['id']] = data['ids']

    def load_samples(self):
        indexer = IndexReader(self.args.index_dir)
        custom_bm25 = search.LuceneSimilarities.bm25(self.args.bm25_k1, self.args.bm25_b)

        if self.mode == 'train':
            # top docs by BM25
            top_lst = defaultdict(list)
            for line in tqdm(open(os.path.join(self.args.msmarco_dir, f"top_candidates.{self.mode}.tsv"), 'r'),
                            desc="top candidates"):
                qid, pid, score = line.split('\t')
                score = score.rstrip()
                top_lst[qid].append({'pid': int(pid), 'score': float(score)})
            top_lst = dict(top_lst)

            # all queries text (for calculating the BM25 scores)
            queries_text = dict()
            for line in tqdm(open(os.path.join(self.args.msmarco_dir, f"queries.{self.mode}.tsv"), 'r'),
                            desc="queries text"):
                qid, text = line.split('\t')
                text = text.rstrip()
                queries_text[qid] = text

            qids, pos_pids, neg_pids, pos_scores, neg_scores = [], [], [], [], []
            for line in tqdm(open(os.path.join(self.args.msmarco_dir, f"qrels.{self.mode}.tsv"), 'r'), 
                            desc="qrels"):
                qid, _, pid, _ = line.split('\t')
                if qid in top_lst:
                    neg_docs = random_sample(top_lst[qid], min(8, len(top_lst[qid])))  #8: number of neg docs selected from top BM25-assessed candidates
                    pos_score = indexer.compute_query_document_score(pid, queries_text[qid], similarity=custom_bm25)

                    for neg_doc in neg_docs:
                        qids.append(qid)
                        pos_pids.append(int(pid))
                        neg_pids.append(neg_doc['pid'])
                        pos_scores.append(pos_score)
                        neg_scores.append(neg_doc['score'])
            self.qids, self.pos_pids, self.neg_pids = qids, pos_pids, neg_pids
            self.pos_scores, self.neg_scores = pos_scores, neg_scores
        else:
            qids, pids, scores = [], [], []
            qrels_path = os.path.join(self.args.msmarco_dir, f"qrels.{self.mode}.tsv")
            if os.path.exists(qrels_path):
                for line in open(qrels_path, 'r'):
                    qid, _, pid, _ = line.split('\t')
                    qids.append(qid)
                    pids.append(int(pid))
                    scores.append(indexer.compute_query_document_score(pid, qid, similarity=custom_bm25))

            for line in open(os.path.join(self.args.msmarco_dir, f"top_candidates.{self.mode}.tsv"), 'r'):
                qid, pid, score = line.split('\t')
                score = score.rstrip()
                qids.append(qid)
                pids.append(int(pid))
                scores.append(score)
            self.qids, self.pids, self.scores = qids, pids, scores


def pack_tensor(lstlst, dtype, default=0, length=None):  #将list形式的batch数据转化为tensor
    batch_size = len(lstlst)
    length = length if length is not None else max(len(l) for l in lstlst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)  #先构建全0向量(即padding), 后补充数据
    for i, l in enumerate(lstlst):
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
    return tensor


def get_collate_function(mode):  #将batch数据转化为tensor
    if mode == 'train':
        def collate_function(batch):  #batch为多个sample的list集
            query_input_ids = [x["query_input_ids"] for x in batch]
            pos_doc_input_ids = [x["pos_doc_input_ids"] for x in batch]
            neg_doc_input_ids = [x["neg_doc_input_ids"] for x in batch]
            query_mask = [[1] * len(input_ids) for input_ids in query_input_ids]
            pos_doc_mask = [[1] * len(input_ids) for input_ids in pos_doc_input_ids]
            neg_doc_mask = [[1] * len(input_ids) for input_ids in neg_doc_input_ids]
            pos_s_lex = [[x["pos_score"]] for x in batch]
            neg_s_lex = [[x["neg_score"]] for x in batch]
            data = {
                "query_input_ids": pack_tensor(query_input_ids, dtype=torch.int64),
                "pos_doc_input_ids": pack_tensor(pos_doc_input_ids, dtype=torch.int64),
                "neg_doc_input_ids": pack_tensor(neg_doc_input_ids, dtype=torch.int64),
                "query_mask": pack_tensor(query_mask, dtype=torch.int64),
                "pos_doc_mask": pack_tensor(pos_doc_mask, dtype=torch.int64),
                "neg_doc_mask": pack_tensor(neg_doc_mask, dtype=torch.int64),
                "pos_s_lex": torch.tensor(pos_s_lex, dtype=torch.float32),
                "neg_s_lex": torch.tensor(neg_s_lex, dtype=torch.float32),
            }
            return data
    else:
        def collate_function(batch):  #batch为多个sample的list集
            query_input_ids = [x["query_input_ids"] for x in batch]
            doc_input_ids = [x["doc_input_ids"] for x in batch]
            query_mask = [[1] * len(input_ids) for input_ids in query_input_ids]
            doc_mask = [[1] * len(input_ids) for input_ids in doc_input_ids]
            s_lex = [[x["score"]] for x in batch]
            data = {
                "query_input_ids": pack_tensor(query_input_ids, dtype=torch.int64),
                "doc_input_ids": pack_tensor(doc_input_ids, dtype=torch.int64),
                "query_mask": pack_tensor(query_mask, dtype=torch.int64),
                "doc_mask": pack_tensor(doc_mask, dtype=torch.int64),
                "s_lex": torch.tensor(s_lex, dtype=torch.float16),
            }
            qid_lst = [x['qid'] for x in batch]
            pid_lst = [x['pid'] for x in batch]
            return data, qid_lst, pid_lst
    return collate_function
