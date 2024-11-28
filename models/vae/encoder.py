import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import typing
from typing import Union


class Encoder(nn.Module):

    def __init__(self):

        super().__init__()
        self.encoder = nn.ModuleList([
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            vae_residual_block(128,128)
            vae_residual_block(128,128),
            nn.Conv2d(128,128,kernel_size=3, stride=2, padding),
            vae_residual_block(128,256),
            vae_residual_block(256,256),
            nn.Conv2d(256,256, kernel_size=3, stride=2, padding=0)
            vae_residual_block(256,512),
            vae_residual_block(512,512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0)
            vae_residual_block(512,512),
            vae_residual_block(512,512),
            vae_residual_block(512,512),
            vae_attention(512),
            vae_residual_block(512,512),
            nn.GroupNorm(32, 512),
            nn.SILU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8,8, kernel_size=1, padding=0)]
        )

    def forward(self, x: Tensor, noise: Tensor) -> Tensor:
        for layer in self.encoder:
            if layer.stride == (2,2):
                x = F.pad(x, (0,1,0,1)
            x = layer(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        x = mean + stdev * noise
        x *= 0.18125
        return x
