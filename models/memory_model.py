import torch 
from torch import nn
from ..utils import mean_pooling
from attention import ScaledDotProductAttention
from transformers import AutoModel

class MemoryModel(nn.Module):
    
    def __init__(self, model_name_or_path, memory_size=20) -> None:
        super(MemoryModel, self).__init__()
        # basemodel (not trainable)
        self.bartmodel = AutoModel.from_pretrained(model_name_or_path)
        for param in self.bartmodel.parameters():
            param.requires_grad = False
        # memory
        self.memory = nn.ModuleList([nn.Linear(self.bartmodel.config.hidden_size, self.bartmodel.config.hidden_size) for i in range(memory_size)])
        # attention
        self.attn_layer = ScaledDotProductAttention(self.bartmodel.config.hidden_size, self.bartmodel.config.hidden_size)

    def cl_forward(self, query_inputs, positive_inputs):
        with torch.no_grad():
            query_outputs = self.bartmodel(
                input_ids=query_inputs["input_ids"],
                attention_mask=query_inputs["attention_mask"]
            )
            positive_outputs = self.bartmodel(
                input_ids=positive_inputs["input_ids"],
                attention_mask=positive_inputs["attention_mask"]
            )
            query_emb, pos_emb = mean_pooling(query_outputs, query_inputs["attention_mask"]), mean_pooling(positive_outputs, positive_inputs["attention_mask"])
            
        concat_query_embs = torch.stack(
                [layer(query_emb) for layer in self.memory], dim=1
        )
        query_transformed, query_attn_weights = self.attn_layer(query=query_emb.unsqueeze(1), key=concat_query_embs, value=concat_query_embs)
        query_transformed = query_transformed.squeeze(1)
        return {
            "positive_emb": pos_emb,
            "query_transformed": query_transformed, 
            "query_attn_weights": query_attn_weights
        }
    
    def sentemb_forward(self, query_inputs):
        with torch.no_grad():
            query_outputs = self.bartmodel(
                input_ids=query_inputs["input_ids"],
                attention_mask=query_inputs["attention_mask"]
            )
            query_emb = mean_pooling(query_outputs, query_inputs["attention_mask"])
        concat_query_embs = torch.stack(
                [layer(query_emb) for layer in self.memory], dim=1
        )
        query_transformed, query_attn_weights = self.attn_layer(query=query_emb.unsqueeze(1), key=concat_query_embs, value=concat_query_embs)
        query_transformed = query_transformed.squeeze(1)
        return {
            "query_transformed": query_transformed, 
            "query_attn_weights": query_attn_weights
        }
    def forward(self, query_inputs, positive_inputs=None):
        if positive_inputs:
            return self.cl_forward(query_inputs, positive_inputs)
        return self.sentemb_forward(query_inputs)