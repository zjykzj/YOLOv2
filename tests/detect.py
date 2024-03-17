# -*- coding: utf-8 -*-

"""
@Time    : 2024/3/17 19:54
@File    : yolov2_dddd.py
@Author  : zj
@Description: 
"""

import yaml

from models.yolo import YOLOv2Detect
from models.yolo import Detect, Model

anchors = [[12.995, 15.398, 25.905, 37.697, 62.199, 62.933, 107.29, 148.17, 293.6, 281.88]]


def test_yolov2_detect():
    model = YOLOv2Detect(nc=80, anchors=anchors, ch=(30,))
    model.stride = [32, ]
    print(model)

    import torch

    model.train()
    res = model([torch.randn(10, 30, 20, 20)])
    print(f"type(res): {type(res)} - len(res): {len(res)}")
    print(f"res[0].shape: {res[0].shape}")

    print('*' * 100)
    model.eval()
    output, res = model([torch.randn(10, 30, 20, 20)])
    print(f"type(res): {type(res)} - len(res): {len(res)}")
    print(f"res[0].shape: {res[0].shape}")

    print(f"type(output): {type(output)} - len(output): {len(output)}")
    print(f"output.shape: {output.shape}")

    print('*' * 100)
    model.export = True
    output = model([torch.randn(10, 30, 20, 20)])[0]
    print(f"type(output): {type(output)} - len(output): {len(output)}")
    print(f"output.shape: {output.shape}")


def test_yolov5_detect():
    model = Detect(nc=80, anchors=anchors, ch=(30,))
    model.stride = [32, ]
    print(model)

    import torch

    model.train()
    res = model([torch.randn(10, 30, 20, 20)])
    print(f"type(res): {type(res)} - len(res): {len(res)}")
    print(f"res[0].shape: {res[0].shape}")

    print('*' * 100)
    model.eval()
    output, res = model([torch.randn(10, 30, 20, 20)])
    print(f"type(res): {type(res)} - len(res): {len(res)}")
    print(f"res[0].shape: {res[0].shape}")

    print(f"type(output): {type(output)} - len(output): {len(output)}")
    print(f"output.shape: {output.shape}")

    print('*' * 100)
    model.export = True
    output = model([torch.randn(10, 30, 20, 20)])[0]
    print(f"type(output): {type(output)} - len(output): {len(output)}")
    print(f"output.shape: {output.shape}")


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


def create_yolov2detect_model():
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


if __name__ == '__main__':
    # test_yolov2_detect()
    # test_yolov5_detect()
    yolov2_model = create_yolov2detect_model()
    pred, targets = create_pred_targets(yolov2_model)

