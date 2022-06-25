
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, Tuple

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: in_features, out_featurres
        in_features (int): dimention of input features
        out_featurres (int): dimention of input features

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn_weights
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn_weights**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, in_features, out_features):
        super(ScaledDotProductAttention, self).__init__()
        self.q_proj = nn.Linear(in_features, out_features)
        self.k_proj = nn.Linear(in_features, out_features)
        self.v_proj = nn.Linear(in_features, out_features)
        self.out_proj = nn.Linear(out_features, out_features)
        

    def forward(self, 
            query: Tensor, 
            key: Tensor, 
            value: Tensor, 
            mask: Optional[Tensor] = None
        ) -> Tuple[Tensor, Tensor]:
        
        q_out = self.q_proj(query)
        k_out = self.k_proj(key)
        v_out = self.v_proj(value)

        d_model = q_out.size()[-1]
        attn_logits = torch.bmm(q_out, k_out.transpose(1,2)) / np.sqrt(d_model)
        
        if mask is not None:
            attn_logits.masked_fill_(mask.view(attn_logits.size()), -float('Inf'))

        attn_weights = F.softmax(attn_logits, -1)
        context = torch.bmm(attn_weights, v_out)
        return context, attn_weights