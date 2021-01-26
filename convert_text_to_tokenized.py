import os
import json
import argparse
from tqdm import tqdm
from transformers import BertTokenizerFast

def tokenize_file(tokenizer, input_file, output_file):
    total_size = sum(1 for _ in open(input_file))
    with open(output_file, 'w') as outFile:
        for line in tqdm(open(input_file), total=total_size,
                desc=f"Tokenize: {os.path.basename(input_file)}"):
            seq_id, text = line.split("\t")
            ids = tokenizer.encode(text, add_special_tokens=False)
            outFile.write(json.dumps(
                {"id": seq_id, "ids":ids}
            ))
            outFile.write("\n")
    
def tokenize_queries(args, tokenizer):
    for mode in ["dev", "train"]:  #["eval.small", "dev", "eval", "train"]
        query_output = f"{args.output_dir}/queries.{mode}.json"
        tokenize_file(tokenizer, f"{args.msmarco_dir}/queries.{mode}.tsv", query_output)

def tokenize_collection(args, tokenizer):
    collection_output = f"{args.output_dir}/collection.tokenize.json"
    tokenize_file(tokenizer, f"{args.msmarco_dir}/collection.tsv", collection_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--msmarco_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./data/tokenize")
    parser.add_argument("--tokenizer_path", type=str, default="bert-base-uncased")
    parser.add_argument("--queries", action="store_true")
    parser.add_argument("--collection", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path)

    if args.queries:
        tokenize_queries(args, tokenizer)  
    if args.collection:
        tokenize_collection(args, tokenizer) 
