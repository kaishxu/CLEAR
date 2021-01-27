import os
import gc
import json
import argparse
from tqdm import tqdm

def compress(args, mode='train'):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    top_lst = dict()
    top_query_lst = dict()
    count = 0
    print('Start loading...')
    with open(os.path.join(args.msmarco_dir, f'top1000.{mode}'), 'r') as f:
        while True:
            if count % 1000000 == 0:
                print("Samples:", count)
            line = f.readline()  # 逐行读取
            if not line:
                break
            qid, pid, query, passage = line.split('\t')
            if qid not in top_lst:
                top_lst[qid] = list()
                top_query_lst[qid] = query

            top_lst[qid].append(int(pid))
            count += 1

    gc.collect()  #清内存
    print('Writing...')
    with open(os.path.join(args.output_dir, f'top1000.{mode}.json'), 'w') as outfile:
        json.dump(top_lst, outfile)
    with open(os.path.join(args.output_dir, f'top1000.{mode}.queries.json'), 'w') as outfile:
        json.dump(top_query_lst, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--msmarco_dir', type=str, default="./data")
    parser.add_argument('--output_dir', type=str, default="./data")
    args = parser.parse_args()

    compress(args, 'train')
