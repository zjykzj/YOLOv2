# -*- coding: utf-8 -*-

"""
@Time    : 2024/4/16 15:41
@File    : tttest.py
@Author  : zj
@Description: 
"""

# import torch
#
# bs = 16
#
# targets = torch.tensor([
#     [1, 2, 3, 4, 5, 6],
#     [1, 3, 3, 4, 5, 6],
#     [2, 4, 3, 4, 351, 16],
#     [2, 10, 3, 23, 5, 6],
# ])
# print(targets.shape)
#
# new_targets = torch.zeros(bs, 50, 5)
# print(new_targets.shape)
#
# for bi in range(bs):
#     if torch.sum(targets[..., 0] == bi) > 0:
#         sub_targets = targets[targets[..., 0] == bi]
#         new_targets[bi][:sub_targets.size(0)] = sub_targets[..., 1:]
#
# print(new_targets)

import numpy as np

x = np.array([[45.726, 60.759],
              [95.308, 136.83],
              [254.51, 174.04],
              [190.07, 325.79],
              [453.31, 371.43], ])

print(x.reshape(-1))
