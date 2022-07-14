import torch 
from torch import nn
from ..utils import mean_pooling
from models.Attention import ScaledDotProductAttentionForMemory

from transformers import AutoModel


cos = nn.CosineSimilarity(dim=1, eps=1e-6)
device = torch.device("cuda" if torch.cuda.is_availabel() else "cpu")

class MemoryModel(nn.Module):
    def __init__(self):
        super(MemoryModel, self).__init__()
        self.bart = AutoModel.from_pretrained("vinai/bartpho-word")
        for param in self.bart.parameters():
            param.requires_grad = False
        self.attn = ScaledDotProductAttentionForMemory(self.bart.config.hidden_size, self.bart.config.hidden_size)
        self.linear = nn.Linear(self.bart.config.hidden_size, self.bart.config.hidden_size)

    def forward(self, query_feats, query_label_feats, retr_feats, gt_feats):
        input_keys = ["input_ids", "attention_mask"]

        query_features = query_feats
        query_label_features = query_label_feats
        bm25_retr_features = retr_feats
        bm25_gt_features = gt_feats
        
        query_inputs = {key: query_features[key].view(-1, query_features[key].shape[-1]).to(device) for key in input_keys}
        query_output = self.bart(**query_inputs)
        query_emb = mean_pooling(query_output, query_inputs["attention_mask"]) # [1, 1024]

        query_label_inputs = {key: query_label_features[key].view(-1, query_label_features[key].shape[-1]).to(device) for key in input_keys}
        query_label_output = self.bart(**query_label_inputs)
        query_label_emb = mean_pooling(query_label_output, query_label_inputs["attention_mask"]) # [1, 1024]

        bm25_retr_inputs = {key: bm25_retr_features[key].view(-1, bm25_retr_features[key].shape[-1]).to(device) for key in input_keys}
        bm25_retr_output = self.bart(**bm25_retr_inputs)
        bm25_retr_emb = mean_pooling(bm25_retr_output, bm25_retr_inputs["attention_mask"]) # [num_sent, 1024]

        bm25_gt_inputs = {key: bm25_gt_features[key].view(-1, bm25_gt_features[key].shape[-1]).to(device) for key in input_keys}
        bm25_gt_output = self.bart(**bm25_gt_inputs)
        bm25_gt_emb = mean_pooling(bm25_gt_output, bm25_gt_inputs["attention_mask"]) # [num_sent, 1024]

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
