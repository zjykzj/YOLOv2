# -*- coding: utf-8 -*-

"""
@date: 2023/1/7 下午12:26
@file: darknet.py
@author: zj
@description: 
"""

import torch
from torch import nn


def conv_bn_act(in_channels: int,
                out_channels: int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                is_bn=True,
                act='relu'):
    # 定义卷积层
    conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    # 定义归一化层
    if is_bn:
        norm = nn.BatchNorm2d(num_features=out_channels)
    else:
        norm = nn.Identity()

    # 定义激活层
    if 'relu' == act:
        activation = nn.ReLU(inplace=True)
    else:
        activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # 返回一个 nn.Sequential 对象，按顺序组合卷积层、归一化层和激活层
    return nn.Sequential(
        conv,
        norm,
        activation
    )


class Backbone(nn.Module):
    cfg = {
        'layer0': [[32, 3]],
        'layer1': ['M', [64, 3]],
        'layer2': ['M', [128, 3], [64, 1], [128, 3]],
        'layer3': ['M', [256, 3], [128, 1], [256, 3]],
        'layer4': ['M', [512, 3], [256, 1], [512, 3], [256, 1], [512, 3]],
        'layer5': ['M', [1024, 3], [512, 1], [1024, 3], [512, 1], [1024, 3]]
    }

    def __init__(self, in_channel=3):
        super(Backbone, self).__init__()
        self.in_channel = in_channel

        self.layer0 = self._make_layers(self.cfg['layer0'])
        self.layer1 = self._make_layers(self.cfg['layer1'])
        self.layer2 = self._make_layers(self.cfg['layer2'])
        self.layer3 = self._make_layers(self.cfg['layer3'])
        self.layer4 = self._make_layers(self.cfg['layer4'])
        self.layer5 = self._make_layers(self.cfg['layer5'])

    def _make_layers(self, layer_cfg):
        layers = []

        # set the kernel size of the first conv block = 3
        for item in layer_cfg:
            if item == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                assert len(item) == 2
                out_channel = item[0]
                kernel_size = item[1]
                padding = 1 if kernel_size == 3 else 0
                layers += conv_bn_act(self.in_channel, out_channel,
                                      kernel_size=kernel_size, stride=1,
                                      padding=padding,
                                      bias=False, is_bn=True, act='leaky_relu'),
                self.in_channel = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        # [N, 3, 224, 224]
        x = self.layer0(x)
        # [N, 32, 224, 224]
        x = self.layer1(x)
        # [N, 64, 112, 112]
        x = self.layer2(x)
        # [N, 128, 56, 56]
        x = self.layer3(x)
        # [N, 256, 28, 28]
        x = self.layer4(x)
        # [N, 512, 14, 14]
        x = self.layer5(x)
        # [N, 1024, 7, 7]

        return x


class Darknet19(nn.Module):

    def __init__(self, num_classes=1000):
        super(Darknet19, self).__init__()
        self.num_classes = num_classes

        self.backbone = Backbone()
        self.fc = nn.Sequential(
            conv_bn_act(1024, num_classes, kernel_size=1, stride=1, padding=0, bias=False, is_bn=True,
                        act='leaky_relu'),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x).view(-1, self.num_classes)
        return x


if __name__ == '__main__':
    m = Darknet19()
    m.eval()

    data = torch.randn(1, 3, 224, 224)
    output = m(data)
    print(output.shape)
