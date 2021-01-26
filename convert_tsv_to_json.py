import json
from tqdm import tqdm
import os
import argparse

def convert_to_json(args):
    f = open(os.path.join(args.msmarco_dir, 'collection.tsv'), 'r')
    if not os.path.exists(args.index_dir):
        os.makedirs(args.index_dir)

    with open(os.path.join(args.index_dir, 'collection.json'), 'w') as outfile:
        outfile.write('[\n')
        for line in tqdm(f.readlines()):
            doc = dict()
            doc['id'], doc['contents'] = line.split('\t')
            outfile.write(json.dumps(doc))
            outfile.write(',\n')
        outfile.write(']')
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--msmarco_dir', type=str, default="./data")
    parser.add_argument('--index_dir', type=str, default="./data/index")
    args = parser.parse_args()

    convert_to_json(args)
