import os
import math
import faiss
import argparse
from tqdm import tqdm
import numpy as np
from utils import generate_rank, eval_results

def get_embeddings(path):
    id_path = os.path.join(path, "ids.memmap")
    id_memmap = np.memmap(id_path, dtype='int32',)
    embedding_path = os.path.join(path, "embedding.memmap")
    embedding_memmap = np.memmap(embedding_path, dtype='float32',  shape=(len(id_memmap), 768))
    return id_memmap, embedding_memmap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute_dir", type=str, default="./data/precompute")
    parser.add_argument("--index_dir", type=str, default="./data/index")
    parser.add_argument("--bm25_dir", type=str, default="./data/bm25_result")
    parser.add_argument("--result_file", type=str, default="./result.tsv")
    parser.add_argument("--search_batch", type=int, default=256)
    args = parser.parse_args()
    query_embedding_dir = os.path.join(args.precompute_dir, "query_embedding")
    doc_embedding_dir = os.path.join(args.precompute_dir, "doc_embedding")

    # get embeddings
    print("Loading embeddings...")
    qids, query_embeddings = get_embeddings(query_embedding_dir)
    pids, doc_embeddings = get_embeddings(doc_embedding_dir)

    # dimension
    assert query_embeddings.shape[1] == doc_embeddings.shape[1]
    dim = query_embeddings.shape[1]

    # initialize faiss
    print("Normalizing...")
    faiss.normalize_L2(query_embeddings)
    faiss.normalize_L2(doc_embeddings)

    # faiss GPU
    print("Initializing FAISS...")
    # res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatIP(dim)
    # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat = index_flat

    # add base vectors
    print("Adding docs...")
    gpu_index_flat.add(doc_embeddings)

    # search
    # num = doc_embeddings.shape[0]
    num = 5000

    steps = math.ceil(query_embeddings.shape[0] / args.search_batch)
    with open(args.result_file, 'w') as outfile:
        for i in tqdm(range(steps), desc="steps"):
            batch = query_embeddings[i * args.search_batch : (i+1) * args.search_batch]
            D, I = gpu_index_flat.search(batch, num)
            for b_i, index_pids in enumerate(I):
                qid = i * args.search_batch + b_i
                for b_j, index_pid in enumerate(index_pids):
                    outfile.write(f"{qids[qid]}\t{pids[index_pid]}\t{D[b_i, b_j]}\n")
