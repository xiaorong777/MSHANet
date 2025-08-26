import os 
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)

import math
import torch
from torch import nn
from utils.util import LinearWithConstraint


#%%
class TalkHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(TalkHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, 

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.25

        # Q, K, V projection
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        #  pre-softmax and post-softmax 
        self.pre_softmax_mix = nn.Parameter(torch.randn(num_heads,num_heads))   # W_pre
        self.post_softmax_mix = nn.Parameter(torch.randn(num_heads,num_heads))  # W_post

    def forward(self, x):
        B, T, D = x.shape

        # 1. Q, K, V 
        qkv = self.qkv_proj(x)  # (B, T, 3D)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, T, head_dim)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)

        # (B, heads, T, T) → (B, T, T, heads)
        attn_logits = attn_logits.permute(0, 2, 3, 1)
        attn_logits = torch.matmul(attn_logits, self.pre_softmax_mix)  # (B, T, T, heads)
        attn_logits = attn_logits.permute(0, 3, 1, 2)  # back to (B, heads, T, T)

        # 4. Softmax
        attn_scores = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_scores)

        # 5. post-softmax 
        attn_weights = attn_weights.permute(0, 2, 3, 1)  # (B, T, T, heads)
        attn_weights = torch.matmul(attn_weights, self.post_softmax_mix)  # (B, T, T, heads)
        attn_weights = attn_weights.permute(0, 3, 1, 2)  # back to (B, heads, T, T)

        context = torch.matmul(attn_weights, v)  # (B, heads, T, head_dim)

        context = context.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(context)
        return out,attn_logits,attn_scores,attn_weights
