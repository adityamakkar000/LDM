import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


import typing


class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed:int, in_proj_bias=True, out_proj_bias=True)

        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads


    def forward(self, x:Tensor, causal_mask: bool = False)


        batch_size, seq_len, d_embed = x.shape
        inter_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        q,k,v = self.in_proj(x).chunk(3,dim=-1)

        q = q.view(inter_shape).transpose(1,2)
        k = k.view(inter_shape).transpose(1,2)
        v = v.view(inter_shape).transpose(1,2)

        weight = q @ k.transpose(-1,-2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=bool).triu(1)
            weight.masked_fill(mask, -torch.inf)

        weight /= self.d_head.sqrt()
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1,2).view(batch_size, seq_len, d_embed)

        output = self.out_proj(output)

        return output













