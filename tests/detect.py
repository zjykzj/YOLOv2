# -*- coding: utf-8 -*-

"""
@Time    : 2024/3/17 19:54
@File    : yolov2_dddd.py
@Author  : zj
@Description: 
"""
from typing import List

import yaml
import torch

from models.yolo import YOLOv2Detect
from models.yolo import Detect, Model

anchors = [[12.995, 15.398, 25.905, 37.697, 62.199, 62.933, 107.29, 148.17, 293.6, 281.88]]


def test_detect():
    def det(model):
        model.train()
        features = model([torch.randn(10, 30, 20, 20)])
        assert isinstance(features, List)
        print(
            f"type(features): {type(features)} - len(features): {len(features)} - features[0].shape: {features[0].shape}")

        print('*' * 100)
        model.eval()
        preds, features = model([torch.randn(10, 30, 20, 20)])
        assert isinstance(features, List)
        assert isinstance(preds, torch.Tensor)
        print(
            f"type(features): {type(features)} - len(features): {len(features)} - features[0].shape: {features[0].shape}")
        print(f"type(preds): {type(preds)} - len(preds): {len(preds)} - preds.shape: {preds.shape}")

        print('*' * 100)
        model.export = True
        preds = model([torch.randn(10, 30, 20, 20)])[0]
        assert isinstance(preds, torch.Tensor)
        print(f"type(preds): {type(preds)} - len(preds): {len(preds)} - preds.shape: {preds.shape}")

    model = YOLOv2Detect(nc=80, anchors=anchors, ch=(30,))
    model.stride = [32, ]
    print(model)
    det(model)

    model = Detect(nc=80, anchors=anchors, ch=(30,))
    model.stride = [32, ]
    print(model)
    det(model)


if __name__ == '__main__':
    test_detect()
