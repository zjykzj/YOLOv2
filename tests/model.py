# -*- coding: utf-8 -*-

"""
@Time    : 2024/3/18 17:36
@File    : model.py
@Author  : zj
@Description: 
"""

from typing import List, Tuple

import yaml
import torch

from models.yolo import Model


def create_data():
    bs = 5
    nc = 80
    img_size = 640
    data = torch.randn(bs, 3, img_size, img_size)

    torch.manual_seed(42)
    # [image_id, class_id, xc, yc, w, h]
    targets = torch.abs(torch.randn((20, 6)) * img_size).to(torch.float32)
    targets[:, 0] = targets[:, 0].int() % bs
    targets[:, 1] = targets[:, 1].int() % nc
    targets[:, 2:] = targets[:, 2:] % img_size / img_size
    print(f"- targets shape: {targets.shape}")
    print(f"- targets type: {type(targets)} dtype: {targets.dtype}")
    print(f"- targets[0]: {targets[0]}\n- targets[-1]: {targets[-1]}")

    return data, targets


def det(data, targets, model):
    assert isinstance(data, torch.Tensor) and isinstance(targets, torch.Tensor)

    model.train()
    pred = model.forward(data)
    assert isinstance(pred, List)
    print(f"[Train] len(pred): {len(pred)} - pred[0].shape: {pred[0].shape}")

    model.eval()
    pred = model.forward(data)
    assert isinstance(pred, Tuple) and len(pred) == 2
    print(f"[Eval Export=False] len(pred): {len(pred)}")
    print(f"- pred[0] shape: {pred[0].shape}")
    for i, pred in enumerate(pred[1]):
        print(f"- pred[1][{i}] shape: {pred[1][i].shape}")

    model.eval()
    model.model[-1].export = True
    pred = model.forward(data)
    assert isinstance(pred, Tuple) and len(pred) == 1
    print(f"[Eval Export=True] len(pred): {len(pred)} - pred[0].shape: {pred[0].shape}")


if __name__ == '__main__':
    data, targets = create_data()
    hyp = '../data/hyps/hyp.scratch-low.yaml'

    model = Model('../models/yolov2v3/yolov2-fast.yaml', ch=3, nc=80, anchors=None)  # create
    model.hyp = hyp
    print(f"- model.stride: {model.stride}\n- Detect.anchors: {model.model[-1].anchors}")
    det(data, targets, model)

    print('*' * 100)

    model = Model('../models/yolov2v3/yolov2-fast_plus.yaml', ch=3, nc=80, anchors=None)  # create
    model.hyp = hyp
    print(f"- model.stride: {model.stride}\n- Detect.anchors: {model.model[-1].anchors}")
    det(data, targets, model)

    print('*' * 100)

    model = Model('../models/yolov5n.yaml', ch=3, nc=80, anchors=None)  # create
    model.hyp = hyp
    print(f"- model.stride: {model.stride}\n- Detect.anchors: {model.model[-1].anchors}")
    det(data, targets, model)
