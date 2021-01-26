# DUALRM
This is the unofficial replication of the paper: [**Replication of Complement Lexical Retrieval with Semantic Residual Embeddings**](https://arxiv.org/abs/2004.13969) (Luyu Gao, Zhuyun Dai). It is still updating! All codes are mainly based on the architecture of [RepBERT](https://github.com/jingtaozhan/RepBERT-Index) ([Jingtao Zhan](https://github.com/jingtaozhan)).

## Dependencies

DUALRM requires the Python 3, Pytorch 1, and [Pyserini](https://github.com/castorini/pyserini/), and uses the [HuggingFace Transformers](https://github.com/huggingface/transformers) library (v4.2.1).

## Data

Before running the model, you need to download the [MS MARCO Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking) dataset at first. The specific datasets are as follows.

1.  [collection.tar.gz](https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz) (all passages);
2. [ queries.tar.gz](https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz) (all queries);
3. [ qrels.train.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv), [ qrels.dev.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv) (all positive labels);
4. [ top1000.train.tar.gz](https://msmarco.blob.core.windows.net/msmarcoranking/top1000.train.tar.gz), [ top1000.dev.tar.gz](https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz) (samples, **provisional**)

All data should be put into the  `./data/` file.

##Preprocessing

To build an indexer by Pyserini, the first step of preprocessing is to create a json-style collection file.

```
python convert_tsv_to_json.py
```

Then using Pyserini to build an indexer.

```
python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
-input ./data \
-index ./data/indexer -storePositions -storeDocvectors -storeRaw
```

The following steps are similar with RepBERT.

```
python convert_text_to_tokenized.py --queries --collection
python convert_collection_to_memmap.py
python compress_top1000.py #the training dataset (still confirming)
```

## Training

If you have done all the preprocessing procedures, please enjoy training!

```
python ./train.py --mode train
```

## Remark

The code is still debugging, since some settings are unclear. I try to contact with the author.

