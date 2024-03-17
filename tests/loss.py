# -*- coding: utf-8 -*-

"""
@Time    : 2024/3/17 20:33
@File    : loss.py
@Author  : zj
@Description: 
"""

import yaml

from utils.yolov2loss import YOLOv2Loss
from utils.loss import ComputeLoss
from models.yolo import Model


def create_model():
    hyp = '../data/hyps/hyp.scratch-low.yaml'
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
        if 'anchors' not in hyp:  # anchors commented in hyp.yaml
            hyp['anchors'] = 3

    model = Model('../models/yolov2-fast.yaml', ch=3, nc=80, anchors=None)  # create
    model.train()
    model.hyp = hyp
    print(f"model.stride: {model.stride}")

    return model


def create_pred_targets(model):
    import torch

    data = torch.randn(5, 3, 640, 640)
    pred = model.forward(data)
    print(f"len(pred): {len(pred)} - pred[0].shape: {pred[0].shape}")

    # pred = torch.randn((5, 85 * 3, 20, 20))
    # print(f"pred shape: {pred.shape}")

    targets = torch.abs(torch.randn((20, 6)) * 640).to(torch.int)
    targets[:, 0] = targets[:, 0] % 5
    targets[:, 1] = targets[:, 1] % 80
    targets[:, 1:] = targets[:, 1:] % 640
    print(f"targets shape: {targets.shape}")
    print(f"targets[0]: {targets[0]} - targets[-1]: {targets[-1]}")

    return pred, targets


def test_yolov5_loss(model, pred, targets):
    compute_loss = ComputeLoss(model)
    print(compute_loss)

    loss, loss_items = compute_loss(pred, targets)
    print(f"loss: {loss} - loss_items: {loss_items}")


def test_yolov2_loss(model, pred, targets):
    compute_loss = YOLOv2Loss(model)
    print(compute_loss)

    loss, loss_items = compute_loss(pred, targets)
    print(f"loss: {loss} - loss_items: {loss_items}")


if __name__ == '__main__':
    model = create_model()
    pred, targets = create_pred_targets(model)

    # test_yolov5_loss(model, pred, targets)
    test_yolov2_loss(model, pred, targets)
