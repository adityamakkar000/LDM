import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from attention import SelfAttention, CrossAttention
import typing


class timeEmbedding(nn.Module):

    def __init__(self):
        super().__init__()

        self.time = nn.Sequential(nn.Linear(n_embed, 4*n_embed),
                                  nn.silu(),
                                  nn.Linear(4 * n_embed, n_embed)
                                  )

    def forward(self, x: Tensor):

        x = self.time(x)
        return x

class SwitchSequntial(nn.Sequential):

    def forward(self, x: Tensor, context: Tensor, time: Tensor) -> Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        return x


class UNET(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.ModuleList([

            SwitchSequntial(
                nn.Conv2d(4, 320, kernel_size=3, padding=1),
                Unet_residualBlock(320, 320),
                UNET_AttnentionBLock(8, 40),
                nn.Conv2d(320,320, kernel_size=3,stride=2, padding=1),
                Unet_residualBlock(320, 640),
                UNET_AttnentionBLock(8, 80),
                UNET_residualblock(640,640),
                UNET_AttentionBLock(8, 80),
                nn.Conv2d(640,640, kernel_size=3, stride=2, padding=1),
                Unet_residualBlock(640, 1280),
                UNET_attnetionBLock(8, 160),
                UNET_residualblock(1280, 1280),
                unet_attentionblock(8, 160),
                nn.Conv2d(1280,1280, kernel_size=3, stride=2, padding=1),
                unet_residualblock(1280,1280),
                unet_residualblock(1280,1280)
        ])

         self.bottleneck = SwitchSequtial(
            unet_residualBlock(1280, 1280),
            unet_attnetionblock(8, 160),
             unet_residualBLock(1280, 1280)
         )


             self.decoders = nn.ModuleList(






                                 )
class diffusion(nn.Module):

    def __init__(self):

        self.time_embedding = timeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(120, 4)

    def forward(self, latent: Tensor, context: Tensor, time: Tensor) -> Tensor:

        time = self.embedding(time)
        output = self.unet(latent, context, time)
        output = self.final(output)

        return output


