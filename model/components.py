# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# https://github.com/megvii-research/NAFNet
# https://github.com/megvii-research/TLC
# ------------------------------------------------------------------------


import torch
import torch.nn as nn


class SimpleGate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleChannelAttention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.convertable = True
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
        )

    def forward(self, x):
        x = self.sca(x) * x
        return x

