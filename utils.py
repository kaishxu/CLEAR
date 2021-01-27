import os
import re
import time
import torch
import random
import numpy as np
from collections import defaultdict
import subprocess
import argparse

def save_model(model, save_name, args):
    save_dir = os.path.join(args.model_save_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(save_dir)
    torch.save(args, os.path.join(save_dir, 'training_args.bin'))


def generate_rank(input_path, output_path):
    score_dict = defaultdict(list)
    for line in open(input_path):
        qid, pid, score = line.split("\t")
        score_dict[int(qid)].append((float(score), int(pid)))
    with open(output_path, "w") as outFile:
        for qid, pid_lst in score_dict.items():
            random.shuffle(pid_lst)
            pid_lst = sorted(pid_lst, key=lambda x:x[0], reverse=True)
            for rank_idx, (score, pid) in enumerate(pid_lst):
                outFile.write("{}\t{}\t{}\n".format(qid, pid, rank_idx + 1))


def eval_results(run_file_path,
        eval_script="./ms_marco_eval.py", 
        qrels="./data/qrels.dev.tsv"):
    assert os.path.exists(eval_script) and os.path.exists(qrels)
    result = subprocess.check_output(['python', eval_script, qrels, run_file_path])
    match = re.search('MRR @10: ([\d.]+)', result.decode('utf-8'))
    mrr = float(match.group(1))
    return mrr


#对随机数生成器设置一个固定的种子, 使实验结果可复现
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def run_parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--mode", choices=["train", "dev", "eval"], required=True)
    parser.add_argument("--output_dir", type=str, default=f"./train")
    parser.add_argument("--bert_path", type=str, default=f"./bert-base-uncased")
    
    parser.add_argument("--msmarco_dir", type=str, default=f"./data")
    parser.add_argument("--collection_memmap_dir", type=str, default="./data/collection_memmap")
    parser.add_argument("--tokenize_dir", type=str, default="./data/tokenize")
    parser.add_argument("--index_dir", type=str, default="./data/index")
    parser.add_argument("--max_query_length", type=int, default=20)
    parser.add_argument("--max_doc_length", type=int, default=256)
    parser.add_argument("--bm25_k1", type=float, default=0.6)
    parser.add_argument("--bm25_b", type=float, default=0.8)
    parser.add_argument("--Lambda", type=float, default=0.1)
    parser.add_argument("--Ksi", type=float, default=1)

    ## Training parameters
    parser.add_argument("--eval_ckpt", type=int, default=None)
    parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int)
    parser.add_argument("--per_gpu_train_batch_size", default=28, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--evaluate_during_training", action="store_true")
    parser.add_argument("--training_eval_steps", type=int, default=5000)

    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--data_num_workers", default=4, type=int)

    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--warmup_steps", default=10000, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=1, type=int)

    args = parser.parse_args()

    time_stamp = time.strftime("%b-%d_%H:%M:%S", time.localtime())
    args.log_dir = f"{args.output_dir}/log/{time_stamp}"
    args.model_save_dir = f"{args.output_dir}/models"
    args.eval_save_dir = f"{args.output_dir}/eval_results"
    return args
