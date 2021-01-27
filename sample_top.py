import os
import argparse
from tqdm import tqdm
from pyserini.search import SimpleSearcher
from collections import defaultdict

def sampling(args, mode):
    # load the positive doc
    qrels = defaultdict(list)
    for line in open(os.path.join(args.msmarco_dir, f"qrels.{mode}.tsv"), 'r'):
        qid, _, pid, _ = line.split('\t')
        qrels[qid].append(int(pid))
    qrels = dict(qrels)

    # load the queries
    queries = dict()
    for line in open(os.path.join(args.msmarco_dir, f"queries.{mode}.tsv"), 'r'):
        qid, query = line.split('\t')
        query = query.rstrip()
        queries[qid] = query

    searcher = SimpleSearcher(args.index_dir)
    searcher.set_bm25(k1=args.bm25_k1, b=args.bm25_b)
    
    with open(os.path.join(args.output_dir, f'top_candidates.{mode}.tsv'), 'w') as outfile:
        for qid in tqdm(qrels):
            query = queries[qid]
            candidates = searcher.search(query, k=args.topN)
            for i in range(len(candidates)):
                outfile.write(f"{qid}\t{candidates[i].docid}\t{candidates[i].score}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', type=str, default="./data/index")
    parser.add_argument('--msmarco_dir', type=str, default="./data")
    parser.add_argument('--output_dir', type=str, default="./data")
    parser.add_argument("--bm25_k1", type=float, default=0.6)
    parser.add_argument("--bm25_b", type=float, default=0.8)
    parser.add_argument("--topN", type=int, default=100)
    args = parser.parse_args()

    # sampling
    sampling(args, mode='train')
    sampling(args, mode='dev')
