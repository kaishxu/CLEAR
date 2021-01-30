import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
import numpy as np

class CLEAR(BertPreTrainedModel):
    def __init__(self, config, args):
        super(CLEAR, self).__init__(config)
        self.bert = BertModel(config)
        self.Ksi = args.Ksi
        self.Lambda = args.Lambda
        self.act = nn.ReLU()
        self.init_weights()
    
    def forward(self, **kwargs):
        if len(kwargs) == 8:
            query_input_ids, query_mask = kwargs['query_input_ids'], kwargs['query_mask']
            pos_doc_input_ids, pos_doc_mask = kwargs['pos_doc_input_ids'], kwargs['pos_doc_mask']
            neg_doc_input_ids, neg_doc_mask = kwargs['neg_doc_input_ids'], kwargs['neg_doc_mask']
            pos_s_lex, neg_s_lex = kwargs['pos_s_lex'], kwargs['neg_s_lex']
            pos_s_emb = self.S_emb(self.encoding(query_input_ids, query_mask), self.encoding(pos_doc_input_ids, pos_doc_mask))
            neg_s_emb = self.S_emb(self.encoding(query_input_ids, query_mask), self.encoding(neg_doc_input_ids, neg_doc_mask))
            mr = self.Ksi - self.Lambda * (pos_s_lex - neg_s_lex)
            return torch.mean(self.act(mr.squeeze() - pos_s_emb + neg_s_emb))
        elif len(kwargs) == 5:
            query_input_ids, query_mask = kwargs['query_input_ids'], kwargs['query_mask']
            doc_input_ids, doc_mask = kwargs['doc_input_ids'], kwargs['doc_mask']
            s_lex = kwargs['s_lex']
            s_emb = self.S_emb(self.encoding(query_input_ids, query_mask), self.encoding(doc_input_ids, doc_mask))
            s_lex = self.Lambda * s_lex
            return s_lex.squeeze() + s_emb

    def mean_pooling(self, sequence_vectors):
        return torch.mean(sequence_vectors, dim=1)

    def encoding(self, input_ids, attention_mask):
        sequence_vectors = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        rep = self.mean_pooling(sequence_vectors)
        return rep

    def S_emb(self, rep_q, rep_d):
        assert rep_q.shape == rep_d.shape
        return torch.mul(rep_q, rep_d).sum(1) / torch.norm(rep_q, dim=1) / torch.norm(rep_d, dim=1)


class CLEAR_Embedding(BertPreTrainedModel):
    def __init__(self, config):
        super(CLEAR_Embedding, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()
    
    def forward(self, **kwargs):
        input_ids, mask = kwargs['input_ids'], kwargs['mask']
        return self.encoding(input_ids, mask)

    def mean_pooling(self, sequence_vectors):
        return torch.mean(sequence_vectors, dim=1)

    def encoding(self, input_ids, attention_mask):
        sequence_vectors = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        rep = self.mean_pooling(sequence_vectors)
        return rep
