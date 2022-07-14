import torch 
from torch import nn
from ..utils import mean_pooling
from models.Attention import ScaledDotProductAttentionForMemory

from transformers import AutoModel


cos = nn.CosineSimilarity(dim=1, eps=1e-6)

class MemoryModel(nn.Module):
    def __init__(self):
        super(MemoryModel, self).__init__()
        self.bart = AutoModel.from_pretrained("vinai/bartpho-word")
        for param in self.bart.parameters():
            param.requires_grad = False
        self.attn = ScaledDotProductAttentionForMemory(self.bart.config.hidden_size, self.bart.config.hidden_size)
        self.linear = nn.Linear(self.bart.config.hidden_size, self.bart.config.hidden_size)

    def forward(self, query_feats, query_label_feats, retr_feats, gt_feats):

        query_output = self.bart(**query_feats)
        query_emb = mean_pooling(query_output, query_feats["attention_mask"]) # [1, 1024]
    
        query_label_output = self.bart(**query_label_feats)
        query_label_emb = mean_pooling(query_label_output, query_label_feats["attention_mask"]) # [1, 1024]

        bm25_retr_output = self.bart(**retr_feats)
        bm25_retr_emb = mean_pooling(bm25_retr_output, retr_feats["attention_mask"]) # [num_sent, 1024]

        bm25_gt_output = self.bart(**gt_feats)
        bm25_gt_emb = mean_pooling(bm25_gt_output, gt_feats["attention_mask"]) # [num_sent, 1024]

        bm25_retr_emb = torch.cat([query_emb, bm25_retr_emb], dim=0)
        bm25_gt_emb = torch.cat([query_emb, bm25_gt_emb], dim=0)

        attn_weights = self.attn(query=query_emb, key=bm25_retr_emb, value=bm25_retr_emb, temperature=0.96) # [1, hidden_size]

        Q_argmax = torch.matmul(attn_weights, bm25_retr_emb) # [num_sent, hidden_size]
        P_argmax = torch.matmul(attn_weights, bm25_gt_emb) # [num_sent, hidden_size]

        #fast weight
        self.linear.train()
        for _ in range(5):
            loss1 = 1 - cos(self.linear(Q_argmax), P_argmax) - cos(self.linear(query_emb), query_emb)
            loss1.backward(retain_graph=True)
            for param in self.linear.parameters():
                param.data -= 0.3*param.grad.data
                param.grad.data.zero_()
        self.linear.eval()

        query_transformed = self.linear(query_emb)
        return query_transformed, query_label_emb
