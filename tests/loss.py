# -*- coding: utf-8 -*-

"""
@Time    : 2024/3/17 20:33
@File    : loss.py
@Author  : zj
@Description: 
"""

from typing import List
import yaml
import torch

from models.yolov2v3.yolov2loss import YOLOv2Loss
from utils.loss import ComputeLoss
from models.yolo import Model

from model import create_data


def test_yolov5_loss():
    data, targets = create_data()
    assert isinstance(data, torch.Tensor) and isinstance(targets, torch.Tensor)

    def create_model():
        hyp = 'data/hyps/hyp.scratch-low.yaml'
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            # if 'anchors' not in hyp:  # anchors commented in hyp.yaml
            #     hyp['anchors'] = 3
        model = Model('models/yolov2v3/yolov2-fast_plus.yaml', ch=3, nc=80, anchors=None)  # create
        model.hyp = hyp
        print(f"model.stride: {model.stride}")

        return model

    model = create_model()
    model.train()
    pred = model.forward(data)
    assert isinstance(pred, List)
    print(f"len(pred): {len(pred)} - pred[0].shape: {pred[0].shape}")

    compute_loss = ComputeLoss(model)
    print(compute_loss)

    loss, loss_items = compute_loss(pred, targets)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(loss_items, torch.Tensor)
    print(f"type(loss): {type(loss)} - loss: {loss} - loss_items: {loss_items}")


def test_yolov2_loss():
    data, targets = create_data()
    assert isinstance(data, torch.Tensor) and isinstance(targets, torch.Tensor)

    def create_model():
        hyp = 'data/hyps/hyp.scratch-low.yaml'
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            # if 'anchors' not in hyp:  # anchors commented in hyp.yaml
            #     hyp['anchors'] = 3
        model = Model('models/yolov2v3/yolov2-fast.yaml', ch=3, nc=80, anchors=None)  # create
        model.hyp = hyp
        print(f"model.stride: {model.stride}")

        return model

    model = create_model()
    model.train()
    pred = model.forward(data)
    assert isinstance(pred, List)
    print(f"len(pred): {len(pred)} - pred[0].shape: {pred[0].shape}")

    compute_loss = YOLOv2Loss(model)
    print(compute_loss)

    loss, loss_items = compute_loss(pred, targets)
    print(f"loss: {loss} - loss_items: {loss_items}")


if __name__ == '__main__':
    test_yolov5_loss()
    test_yolov2_loss()
