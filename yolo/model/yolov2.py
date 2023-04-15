# -*- coding: utf-8 -*-

"""
@date: 2023/4/15 下午4:43
@file: yolov2.py
@author: zj
@description: 
"""

import torch
from torch import nn
import torch.nn.functional as F

from darknet.darknet import conv_bn_act, Darknet19


class ReorgLayer(nn.Module):

    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        # [1, 64, 26, 26]
        N, C, H, W = x.shape[:4]
        ws = self.stride
        hs = self.stride

        # [N, C, H, W] -> [N, C, H/S, S, W/S, S] -> [N, C, H/S, W/S, S, S]
        # [1, 64, 26, 26] -> [1, 64, 13, 2, 13, 2] -> [1, 64, 13, 13, 2, 2]
        x = x.view(N, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        # [N, C, H/S, W/S, S, S] -> [N, C, H/S * W/S, S * S] -> [N, C, S * S, H/S * W/S]
        # [1, 64, 13, 13, 2, 2] -> [1, 64, 13 * 13, 2 * 2] -> [1, 64, 2 * 2, 13 * 13]
        x = x.view(N, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        # [N, C, S * S, H/S * W/S] -> [N, C, S * S, H/S, W/S] -> [N, S * S, C, H/S, W/S]
        # [1, 64, 2 * 2, 13 * 13] -> [1, 64, 2*2, 13, 13] -> [1, 2*2, 64, 13, 13]
        x = x.view(N, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
        # [N, S * S, C, H/S, W/S] -> [N, S * S * C, H/S, W/S]
        # [1, 2*2, 64, 13, 13] -> [1, 2*2*64, 13, 13]
        x = x.view(N, hs * ws * C, int(H / hs), int(W / ws))
        # [1, 256, 13, 13]
        return x


class Backbone(nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()

        darknet19 = Darknet19()
        # darknet backbone
        self.conv1 = nn.Sequential(darknet19.backbone.layer0,
                                   darknet19.backbone.layer1,
                                   darknet19.backbone.layer2,
                                   darknet19.backbone.layer3,
                                   darknet19.backbone.layer4)

        self.conv2 = darknet19.backbone.layer5

    def forward(self, x):
        # x: [1, 3, 416, 416]
        last_x = self.conv1(x)
        # last_x: [1, 512, 26, 26]
        x = self.conv2(last_x)
        # x: [1, 1024, 13, 13]

        return x, last_x


class Head(nn.Module):

    def __init__(self, num_classes=20, num_anchors=5):
        super(Head, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # detection layers
        self.conv3 = nn.Sequential(
            conv_bn_act(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True,
                        act='leaky_relu'),
            conv_bn_act(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True,
                        act='leaky_relu'),
        )

        self.downsampler = conv_bn_act(512, 64, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True,
                                       act='leaky_relu')
        self.reorg = ReorgLayer()

        self.conv4 = nn.Sequential(
            conv_bn_act(1280, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True,
                        act='leaky_relu'),
            nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1)
        )

    def forward(self, x, last_x):
        # last_x: [1, 512, 26, 26]
        last_x = self.downsampler(last_x)
        # last_x: [1, 64, 26, 26]
        last_x = self.reorg(last_x)
        # last_x: [1, 256, 13, 13]

        # x: [1, 1024, 13, 13]
        x = self.conv3(x)
        # x: [1, 1024, 13, 13]

        x = torch.cat([last_x, x], dim=1)
        # x: [1, 1280, 13, 13]
        x = self.conv4(x)
        # x: [1, 125, 13, 13]
        # 通道层 = num_anchors * (5+num_classes) = 5 * (5+20) = 5*25 = 125
        return x


class YOLOLayer(nn.Module):

    def __init__(self):
        super(YOLOLayer, self).__init__()

    def forward(self, out):
        N, C, H, W = out.size()

        # [1, 125, 13, 13] -> [1, 13, 13, 125] -> [1, 13*13*5, 5+20] -> [1, 845, 25]
        out = out.permute(0, 2, 3, 1).contiguous().view(N, H * W * self.num_anchors, 5 + self.num_classes).contiguous()

        # activate the output tensor
        # `sigmoid` for t_x, t_y, t_c; `exp` for t_h, t_w;
        # `softmax` for (class1_score, class2_score, ...)
        # [xc, yc]数值压缩到(0, 1)
        # [N, h*w*num_anchors, 2]
        xy_pred = torch.sigmoid(out[:, :, 0:2])
        # [box_conf]数值压缩到(0, 1)
        # [N, h*w*num_anchors, 1]
        conf_pred = torch.sigmoid(out[:, :, 4:5])
        # [box_h, box_w]数值进行指数运算
        # [N, h*w*num_anchors, 2]
        hw_pred = torch.exp(out[:, :, 2:4])
        # [N, h*w*num_anchors, num_classes]
        class_score = out[:, :, 5:]
        # 计算每个锚点框的分类概率
        class_pred = F.softmax(class_score, dim=-1)
        # [N, h*w*num_anchors, xc+yc+box_w+box_h]
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

        return delta_pred, conf_pred, class_pred


class YOLOv2(nn.Module):

    def __init__(self, num_classes=20, num_anchors=5):
        super(YOLOv2, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.backbone = Backbone()
        self.head = Head()
        self.yolo_layer = YOLOLayer()

    def forward(self, x):
        x, last_x = self.backbone(x)
        x = self.head(x, last_x)

        delta_pred, conf_pred, class_pred = self.yolo_layer(x)
        return delta_pred, conf_pred, class_pred
