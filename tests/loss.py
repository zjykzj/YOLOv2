# -*- coding: utf-8 -*-

"""
@date: 2023/4/27 下午3:37
@file: anchors.py
@author: zj
@description: 
"""

import torch
import torch.nn.functional as F


def test_scale():
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)

    mask = torch.ones(3, 4)
    mask *= 0.5
    mask[1] = 5

    aa = a * mask
    bb = b * mask
    loss1 = F.mse_loss(aa, bb, reduction='sum')

    aa1 = a[mask == 0.5]
    bb1 = b[mask == 0.5]
    loss2 = F.mse_loss(aa1, bb1, reduction='sum')

    aa2 = a[mask == 5]
    bb2 = b[mask == 5]
    loss3 = F.mse_loss(aa2, bb2, reduction='sum')

    assert (loss1 - (loss2 * 0.5 * 0.5 + loss3 * 5 * 5)) < 1e-3

    loss4 = F.mse_loss(aa1 * 0.5, bb1 * 0.5, reduction='sum')
    loss5 = F.mse_loss(aa2 * 5, bb2 * 5, reduction='sum')
    assert (loss1 - (loss4 + loss5)) < 1e-3


if __name__ == '__main__':
    test_scale()
