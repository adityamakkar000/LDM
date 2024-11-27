import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from attention import SelfAttention

import typing


class vae_attention(nn.Module):

    def __init__(self, channels: int):

        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x:Tensor) -> torch.Tensor:

        residual = x

        b,c,h,w = x.shape

        x = x.view(b, c, h*w)
        x = x.transpose(-1, -2)

        x = self.attention(x)


        x = x.transpose(-1, -2)
        x = x.view(b, c, h ,w)

        return x


class vae_residualblock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Idenity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: Tensor) -> Tensor
        residual = self.residual_layer(x)
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        x =  x + residual
        return x

class vae_decoder(nn.Module):

    def __init__(self):

        super().__init__()


        self.decoder = nn.ModuleList([

            nn.Conv2d(4,4,kernel_size=1, padding=1),
            nn.Conv2d(4, 512, kernel_size, 3, padding=0),
            vae_residualblock(512,512),
            vae_residualblock(512,512),
            vae_attention(512,512),
            vae_residualblock(512,512),
            vae_residualblock(512,512),
            vae_residualblock(512,512),
            vae_residualblock(512,512),
            vae_residualblock(512,512),
            nn.Upsample(scale=2),
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            vae_residualblock(512,512),
            vae_residualblock(512,512),
            vae_residualblock(512,512),
            nn.Upsample(scale=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            vae_residualblock(512,256),
            vae_residualblock(256,256),
            vae_residualblock(256,256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,256,kernel_size=3, padding=1),
            vae_residualblock(256, 128),
            vae_residualblock(128,128),
            vae_residualblock(128,128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128,3,kernel_size=3, padding=1)
        ])

    def forward(self, x):

        x /= 0.18215

        for layer in self.decoder:
            x = layer(x)

        return x








