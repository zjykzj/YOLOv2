# -*- coding: utf-8 -*-

"""
@Time    : 2024/4/5 22:34
@File    : components.py
@Author  : zj
@Description: 
"""

import torch.nn as nn
from models.common import Conv


class ResBlock(nn.Module):

    def __init__(self, ch, num_blocks=1, shortcut=True):

        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(num_blocks):
            self.module_list.append(nn.Sequential(
                # 1x1卷积，通道数减半，不改变空间尺寸
                Conv(ch, ch // 2, 1, 1, 0),
                # 3x3卷积，通道数倍增，恢复原始大小，不改变空间尺寸
                Conv(ch // 2, ch, 3, 1, 1)
            ))

    def forward(self, x):
        for module in self.module_list:
            h = x
            h = module(h)
            x = x + h if self.shortcut else h
        return x
