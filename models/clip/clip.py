import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from attention import SelfAttention
import typing


class ClIPEMbedding(nn.Module):

    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):

        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Paramter(torch.zeros(n_vocab, n_embed))

    def forward(self, token: Tensor) -> Tensor:

        x = self.token_embedding(token) + self.position_embedding
        return x

class CLIPLayer(nn.Module):

    def __init__(self, n_embed, n_head):

        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: Tensor) -> Tensor

        x = self.attention(self.layernorm_1(x)) + x
        residual = x
        x = self.linear_1(self.linear_1(x))
        x *= torch.sigmoid(1.702 * x)
        x = self.linear_2(x) + residual

        return x


class CLIP(nn.Module):

    def __init__(self):


        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([

            CLIPLayer(12, 768) for i in range(12)

        ])
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: Tensor) -> Tensor:

        x = self.embedding(tokens)

        for layer in self.layer:
            x = layer(x)

        output = self.layernorm(state)
        return output




