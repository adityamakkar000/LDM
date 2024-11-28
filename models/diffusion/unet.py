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

class UNET_ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, n_time=1280):

        self.groupnorm_feature = nn.GroupNorm(32, in_chhanels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()

        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: Tensor, time: Tensor) -> Tensor:

            feature = x
            x = self.conv_feature(F.silu(self.groupnorm_feature(x)))
            time = self.linear_time(F.silu(time)).view(time.shape,1,1)
            x =  feature + time
            x = self.conv_feature(F.silu(self.groupnorm_merged(x))) + self.residual_layer(feature)
            return x

class UNET_AttentionBlobkck(nn.Module):

    def __init__(self, n_heads, n_embed):

        super().__init__()

        channels = n_heads * n_embed
        self.groupnorm = nn.GroupNorm(32, chanels)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_2 = nn.Linear(4 * channels, channels)


        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    def forward(self, x:Tensor) -> Tensor:


        residual = x
        x = self.conv_input(self.groupnorm(x))

        n,c,h,w = x.shape
        x = x.view(n,c, h*w).tranpose(-1,-2)
        x = self.attention_1(self.layernorm_1(x)) + x
        x = self.attention_2(self.layernrom_2(x)) + x

        x = self.layernorm_3
        x, g= self.linear_1(x).chunk(2, dim=-1)
        x = x * self.gelu(g)
        x = self.linear_2(x)

        x = self.conv_output(x) + residual
        return x



class Upsample(nn.Module):

    def __init__(self, channels: int):

        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x:Tensor) -> Tensor

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)

class UNET(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])





class UNET_OutputLayer(nn.Module):

    def __init__(self, in_chnnaels: int, out_channels: out):

        self.layer = nn.Sequential(nn.GroupNorm(32, in_channels),
                                   nn.Silu(),
                                   nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3)
                                   )

    def forward(self, x: Tensor) -> Tensor:

        x = self.layer(x)

        return x


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


