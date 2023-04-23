# -*- coding: utf-8 -*-

"""
@date: 2023/4/15 下午4:43
@file: yolov2.py
@author: zj
@description: 
"""
from collections import OrderedDict
import os

from numpy import ndarray

import torch
from torch import Tensor
from torch import nn

from darknet.darknet import conv_bn_act, Darknet19, FastDarknet19

from yolo.util import logging

logger = logging.get_logger(__name__)


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
        x = x.view(N, hs * ws * C, int(H / hs), int(W / ws)).contiguous()
        # [1, 256, 13, 13]
        return x


class Backbone(nn.Module):

    def __init__(self, arch='Darknet19', pretrained=None):
        super(Backbone, self).__init__()
        self.arch = arch

        if 'Darknet19' == arch:
            self.darknet = Darknet19()
        elif 'FastDarknet19' == arch:
            self.darknet = FastDarknet19()
        else:
            raise ValueError(f"{arch} doesn't supports")
        # # darknet backbone
        # self.conv1 = nn.Sequential(darknet19.backbone.layer0,
        #                            darknet19.backbone.layer1,
        #                            darknet19.backbone.layer2,
        #                            darknet19.backbone.layer3,
        #                            darknet19.backbone.layer4)
        #
        # self.conv2 = darknet19.backbone.layer5

        self._init_weights(pretrained=pretrained)

    def _init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)

        if pretrained is not None:
            assert os.path.isfile(pretrained), pretrained
            logger.info(f'Loading pretrained {self.arch}: {pretrained}')

            state_dict = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # strip the names

            self.darknet.load_state_dict(state_dict, strict=True)

    def conv1(self, x):
        x = self.darknet.backbone.layer0(x)
        x = self.darknet.backbone.layer1(x)
        x = self.darknet.backbone.layer2(x)
        x = self.darknet.backbone.layer3(x)
        x = self.darknet.backbone.layer4(x)

        return x

    def conv2(self, x):
        x = self.darknet.backbone.layer5(x)
        return x

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

        self.down_sample = conv_bn_act(512, 64, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True,
                                       act='leaky_relu')
        self.reorg = ReorgLayer()

        self.conv3 = nn.Sequential(
            conv_bn_act(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True,
                        act='leaky_relu'),
            conv_bn_act(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True,
                        act='leaky_relu'),
        )

        # detection layers
        self.conv4 = nn.Sequential(
            conv_bn_act(1280, 1024, kernel_size=3, stride=1, padding=1, bias=False, is_bn=True,
                        act='leaky_relu'),
            nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, last_x):
        # last_x: [1, 512, 26, 26]
        last_x = self.down_sample(last_x)
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

    def __init__(self, anchors, num_anchors=5, num_classes=20, target_size=416, stride=32):
        super(YOLOLayer, self).__init__()
        assert isinstance(anchors, Tensor)
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.target_size = target_size
        self.stride = stride

        F_size = target_size // stride
        # [F_H, F_W]
        self.shift_y, self.shift_x = torch.meshgrid([torch.arange(0, F_size), torch.arange(0, F_size)])
        assert tuple(self.anchors.shape) == (self.num_anchors, 2)
        # [num_anchors, 2] -> [1, 1, num_anchors, 2] -> [F_H, F_W, num_anchors, 2] -> [1, F_H, F_W, num_anchors, 2]
        self.all_grid_anchors = \
            self.anchors.view(1, 1, self.num_anchors, 2).expand(F_size, F_size, self.num_anchors, 2).unsqueeze(0)

    def forward(self, x):
        N, C, H, W = x.shape[:4]

        # [N, C, H, W] -> [N, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        # [N, H, W, C] -> [N, H, W, num_anchors*4] -> [N, H, W, num_anchors, 4]
        pred_box_deltas = x[..., :self.num_anchors * 4].reshape(N, H, W, self.num_anchors, 4)
        # [N, H, W, C] -> [N, H, W, num_anchors]
        pred_confs = x[..., self.num_anchors * 4:self.num_anchors * 5]
        # [N, H, W, C] -> [N, H, W, num_classes]
        pred_cls_probs = x[..., self.num_anchors * 5:]

        # 坐标转换
        # b_x = sigmoid(t_x) + c_x
        # b_y = sigmoid(t_y) + c_y
        # b_w = p_w * e^t_w
        # b_h = p_h * e^t_h
        #
        pred_box_deltas[..., :2] = torch.sigmoid(pred_box_deltas[..., :2])
        # [B, F_H, F_W, num_anchors] + []
        pred_box_deltas[..., 0] += self.shift_x
        pred_box_deltas[..., 1] += self.shift_y
        # [B, F_H, F_W, num_anchors, 2] * [1, F_H, F_W, num_anchors, 2] -> [B, F_H, F_W, num_anchors, 2]
        pred_box_deltas[..., 2:] = torch.exp(pred_box_deltas[..., 2:]) * self.all_grid_anchors
        # 分类概率压缩
        pred_cls_probs = torch.softmax(pred_cls_probs, dim=-1)

        return pred_box_deltas, pred_confs, pred_cls_probs


class YOLOv2(nn.Module):

    def __init__(self, anchors, target_size=416,
                 num_classes=20, num_anchors=5, arch='Darknet19', pretrained=None):
        super(YOLOv2, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.backbone = Backbone(arch=arch, pretrained=pretrained)
        self.head = Head(num_classes=num_classes, num_anchors=num_anchors)
        self.yolo_layer = YOLOLayer(anchors, target_size=target_size,
                                    num_anchors=num_anchors, num_classes=num_classes)

    def forward(self, x):
        x, last_x = self.backbone(x)
        x = self.head(x, last_x)

        if self.training:
            # [B, num_anchors * (5 + num_classes), F_H, F_W]
            return x
        else:
            return self.yolo_layer(x)
