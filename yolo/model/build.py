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

    model_type = cfg['MODEL']['TYPE']
    if 'YOLOv2' == model_type:
        anchors = torch.FloatTensor(cfg['MODEL']['ANCHORS'])
        model = YOLOv2(anchors,
                       num_classes=cfg['MODEL']['N_CLASSES'],
                       arch=cfg['MODEL']['BACKBONE'],
                       pretrained=cfg['MODEL']['BACKBONE_PRETRAINED']
                       ).to(device)
    else:
        raise ValueError(f"{model_type} doesn't supports")

    model = model.to(memory_format=memory_format, device=device)
    return model


def build_criterion(cfg: Dict, device=None):
    loss_type = cfg['CRITERION']['TYPE']
    if 'YOLOv2Loss' == loss_type:
        anchors = torch.FloatTensor(cfg['MODEL']['ANCHORS'])
        criterion = YOLOv2Loss(anchors,
                               num_classes=cfg['MODEL']['N_CLASSES'],
                               ignore_thresh=cfg['CRITERION']['IGNORE_THRESH'],
                               coord_scale=cfg['CRITERION']['COORD_SCALE'],
                               noobj_scale=cfg['CRITERION']['NOOBJ_SCALE'],
                               obj_scale=cfg['CRITERION']['OBJ_SCALE'],
                               class_scale=cfg['CRITERION']['CLASS_SCALE'],
                               ).to(device)
    else:
        raise ValueError(f"{loss_type} doesn't supports")
    return criterion
