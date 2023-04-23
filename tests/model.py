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


def t_yolov2():
    cfg = 'configs/yolov2_default.cfg'
    with open(cfg, 'r') as f:
        import yaml
        cfg = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    anchors = torch.tensor(cfg['MODEL']['ANCHORS'], dtype=torch.float)
    arch = cfg['MODEL']['BACKBONE']
    pretrained = cfg['MODEL']['BACKBONE_PRETRAINED']
    model = YOLOv2(anchors, arch=arch, pretrained=pretrained)
    print(model)

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
    pred_boxes, pred_confs, pred_cls_probs = model(data.to(device))
    print("pred_boxes:", pred_boxes.shape)
    print("pred_confs:", pred_confs.shape)
    print("pred_cls_probs:", pred_cls_probs.shape)


if __name__ == '__main__':
    t_yolov2()