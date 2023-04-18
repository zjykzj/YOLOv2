# -*- coding: utf-8 -*-

"""
@date: 2022/12/14 上午11:00
@file: build.py
@author: zj
@description:
"""

from argparse import Namespace
from typing import Dict

import torch

from .yolov2 import YOLOv2
from .yololoss import YOLOv2Loss


def build_model(args: Namespace, cfg: Dict, device=None):
    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    model = YOLOv2(num_classes=20, num_anchors=5)
    model = model.to(memory_format=memory_format, device=device)

    return model


def build_criterion(cfg: Dict, device=None):
    criterion = YOLOv2Loss(ignore_thresh=float(cfg['CRITERION']['IGNORE_THRESH'])).to(device)
    return criterion
