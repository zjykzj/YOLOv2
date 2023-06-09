# -*- coding: utf-8 -*-

"""
@date: 2023/4/23 上午10:41
@file: yolov2.py
@author: zj
@description: 
"""

import torch
import numpy as np

from yolo.model.yolov2 import YOLOv2


def load_cfg(cfg_file):
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    return cfg


def test_yolov2(cfg_file):
    cfg = load_cfg(cfg_file)
    print(f"load cfg: {cfg_file}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    anchors = torch.tensor(cfg['MODEL']['ANCHORS'], dtype=torch.float)
    arch = cfg['MODEL']['BACKBONE']
    pretrained = cfg['MODEL']['BACKBONE_PRETRAINED']
    model = YOLOv2(anchors, arch=arch, pretrained=pretrained)
    # print(model)

    model = model.to(device)

    print("=> Train")
    model.train()
    image_size = cfg['TRAIN']['IMGSIZE']
    data = torch.randn(1, 3, image_size, image_size)
    res = model(data.to(device))
    print("res:", res.shape)

    print("=> Test")
    model.eval()
    image_size = cfg['TEST']['IMGSIZE']
    data = torch.randn(1, 3, image_size, image_size)
    outputs = model(data.to(device))
    print("outputs:", outputs.shape)


def test_yolov2_fast(cfg_file):
    cfg = load_cfg(cfg_file)
    print(f"load cfg: {cfg_file}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    anchors = torch.tensor(cfg['MODEL']['ANCHORS'], dtype=torch.float)
    arch = cfg['MODEL']['BACKBONE']
    pretrained = cfg['MODEL']['BACKBONE_PRETRAINED']
    model = YOLOv2(anchors, arch=arch, pretrained=pretrained)
    # print(model)

    model = model.to(device)

    print("=> Train")
    model.train()
    image_size = cfg['TRAIN']['IMGSIZE']
    data = torch.randn(1, 3, image_size, image_size)
    res = model(data.to(device))
    print("res:", res.shape)

    print("=> Test")
    model.eval()
    image_size = cfg['TEST']['IMGSIZE']
    data = torch.randn(1, 3, image_size, image_size)
    outputs = model(data.to(device))
    print("outputs:", outputs.shape)


if __name__ == '__main__':
    import random

    random.seed(10)

    cfg_file = 'tests/model/yolov2.cfg'
    test_yolov2(cfg_file)

    cfg_file = 'tests/model/yolov2-tiny.cfg'
    test_yolov2_fast(cfg_file)
