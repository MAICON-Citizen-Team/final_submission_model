# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# https://github.com/megvii-research/NAFNet
# https://github.com/megvii-research/TLC
# ------------------------------------------------------------------------


import torch
import torch.nn as nn
from .components import SimpleGate, SimpleChannelAttention


class NAFBlock(nn.Module):
    def __init__(self, channel, first_expand=2, second_expand=2):
        super().__init__()

        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)

        self.layer1 = nn.Sequential(
            nn.GroupNorm(1, channel), # nn.LayerNorm2D
            nn.Conv2d(channel, channel * first_expand, kernel_size=1), 
            nn.Conv2d(channel * first_expand, channel * first_expand, kernel_size=3, padding=1, groups=channel * first_expand), 
            SimpleGate(), 
            SimpleChannelAttention(channel * first_expand // 2), 
            nn.Conv2d(channel * first_expand // 2, channel, kernel_size=1)
        )
        self.layer2 = nn.Sequential(
            nn.GroupNorm(1, channel), # nn.LayerNorm2D
            nn.Conv2d(channel, channel * second_expand, kernel_size=1), 
            SimpleGate(), 
            nn.Conv2d(channel * second_expand // 2, channel, kernel_size=1)
        )

    def forward(self, x):
        residual = x
        x = self.layer1(x)
        x = x * self.beta + residual

        residual = x
        x = self.layer2(x)
        x = x * self.gamma + residual

        return x


class Encoder(nn.Module):
    def __init__(self, blocks, width):
        super().__init__()
        self.width = width
        self.encoder_blocks = self._build_encoder(blocks)
        self.downsample_blocks = self._build_downsample(blocks)

    def forward(self, x):
        xs = []
        temp = x

        for i in range(len(self.encoder_blocks)):
            xs.append(self.encoder_blocks[i](temp))
            temp = self.downsample_blocks[i](xs[-1])

        xs.append(temp)
        return xs

    def _build_encoder(self, blocks):
        encoder_blocks = nn.ModuleList()

        for i in range(len(blocks)):
            channel = self.width * (2 ** i)
            block = nn.Sequential(*[NAFBlock(channel) for _ in range(blocks[i])])
            encoder_blocks.append(block)

        return encoder_blocks

    def _build_downsample(self, blocks):
        downsample_blocks = nn.ModuleList()

        for i in range(len(blocks)):
            channel = self.width * (2 ** i)
            block = nn.Conv2d(channel, channel * 2, kernel_size=2, stride=2)
            downsample_blocks.append(block)

        return downsample_blocks


class Decoder(nn.Module):
    def __init__(self, blocks, width):
        super().__init__()
        
        self.width = width
        self.decoder_blocks = self._build_decoder(blocks)
        self.upsample_blocks = self._build_upsample(blocks)

    def forward(self, xs):
        ys = []
        temp = xs.pop()

        for i in range(len(self.decoder_blocks)):
            temp = self.upsample_blocks[i](temp) + xs[-i - 1]
            temp = self.decoder_blocks[i](temp)
            ys.append(temp) 

        return ys

    def _build_decoder(self, blocks):
        decoder_blocks = nn.ModuleList()

        for i in range(len(blocks)):
            channel = self.width * (2 ** (len(blocks) - i - 1))
            block = nn.Sequential(*[NAFBlock(channel) for _ in range(blocks[i])])
            decoder_blocks.append(block)

        return decoder_blocks

    def _build_upsample(self, blocks):
        upsample_blocks = nn.ModuleList()

        for i in range(len(blocks)):
            channel = self.width * (2 ** (len(blocks) - i))
            block = nn.Sequential(
                nn.Conv2d(channel, channel * 2, kernel_size=1, bias=False),
                nn.PixelShuffle(2)
            )
            upsample_blocks.append(block)

        return upsample_blocks


class NAFNet(nn.Module):
    def __init__(self, blocks, width):
        super().__init__()

        self.first_layer = nn.Conv2d(3, width, kernel_size=3, padding=1)

        self.encoder = Encoder(blocks["encoder"], width)
        self.center = nn.Sequential(*[NAFBlock(width * (2 ** (len(blocks) + 1)))] * blocks["center"])
        self.decoder = Decoder(blocks["decoder"], width)

        self.last_layer = nn.Conv2d(width, 3, kernel_size=3, padding=1)

    def forward(self, x):
        in_x = x
        x = self.first_layer(x)

        x = self.encoder(x)
        x.append(self.center(x.pop()))
        x = self.decoder(x)

        x = self.last_layer(x[-1])
        out_x = in_x + x

        return out_x
