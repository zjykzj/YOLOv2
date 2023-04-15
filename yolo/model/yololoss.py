# -*- coding: utf-8 -*-

"""
@date: 2023/4/15 下午4:44
@file: yololoss.py
@author: zj
@description: 
"""

from torch import nn


class YOLOLoss(nn.Module):

    def __init__(self):
        super(YOLOLoss, self).__init__()
