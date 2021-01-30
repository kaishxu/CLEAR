import os
import math
import argparse
import pandas as pd
from tqdm import tqdm
from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher
parser = argparse.ArgumentParser()
parser.add_argument('--msmarco_dir', type=str, default="./data")
parser.add_argument('--index_dir', type=str, default="./data/index")
parser.add_argument('--output_dir', type=str, default="./data/bm25_result")
parser.add_argument('--bm25_k1', type=float, default=0.6)
parser.add_argument('--bm25_b', type=float, default=0.8)
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--sample', type=int, default=0)
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

indexer = IndexReader(args.index_dir)
searcher = SimpleSearcher(args.index_dir)
searcher.set_bm25(k1=args.bm25_k1, b=args.bm25_b)
num_candidates = indexer.stats()['documents']

def calculate_bm25(query):
    qid, text = query
    with open(os.path.join(args.output_dir, f"{qid}.tsv"), 'w') as outfile:
        candidates = searcher.search(text, k=num_candidates)
        for i in range(len(candidates)):
            outfile.write(f"{candidates[i].docid}\t{candidates[i].score}\n")

if __name__ == "__main__":
    # load the queries
    queries = dict()
    for line in open(os.path.join(args.msmarco_dir, f"queries.dev.tsv"), 'r'):
        qid, query = line.split('\t')
        query = query.rstrip()
        queries[qid] = query

    qid_lst = set()
    data = pd.read_csv(os.path.join(args.msmarco_dir, "qrels.dev.small.tsv"), sep='\t', header=None)
    
    num = math.ceil(len(data) / 4)  #4 threads
    start = args.sample * num
    end = max((args.sample + 1) * num, len(data))
    for i in tqdm(range(start, end)):
        qid, pid = data.iloc[i, 0].values, data.iloc[i, 2].values
        if qid not in qid_lst:
            qid_lst.add(qid)
            calculate_bm25((qid, queries[str(qid)]))
