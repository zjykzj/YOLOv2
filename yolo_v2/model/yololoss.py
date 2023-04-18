# -*- coding: utf-8 -*-

"""
@date: 2023/4/15 下午4:44
@file: yololoss.py
@author: zj
@description: 
"""

from torch import nn


class YOLOv2Loss(nn.Module):

    def __init__(self):
        super(YOLOv2Loss, self).__init__()



    def forward(self, outputs, targets):
        pass
