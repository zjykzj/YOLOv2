# -*- coding: utf-8 -*-

"""
@date: 2024/4/14 下午4:23
@file: metrics.py
@author: zj
@description: 
"""

import torch
import cv2
import numpy as np

from models.yolov2v3.yolov2loss import bboxes_iou

# 生成三组边界框
boxes1 = torch.tensor([[50.0, 50.0, 150.0, 150.0], [200.0, 200.0, 300.0, 300.0]])  # 有交集
boxes2 = torch.tensor([[100.0, 100.0, 200.0, 200.0], [250.0, 250.0, 350.0, 350.0]])  # 有交集
boxes3 = torch.tensor([[400.0, 400.0, 500.0, 500.0], [600.0, 600.0, 700.0, 700.0]])  # 完全不重叠

# 计算 IoU
iou_matrix1 = bboxes_iou(boxes1, boxes2)
iou_matrix2 = bboxes_iou(boxes1, boxes3)
iou_matrix3 = bboxes_iou(boxes2, boxes3)

print(f"iou_matrix1: {iou_matrix1} - shape: {iou_matrix2.shape}")

# 绘制边界框和交集
image = np.zeros((500, 500, 3), dtype=np.uint8)

# 绘制boxes1和boxes2的边界框
for box in boxes1:
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

for box in boxes2:
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

# 找到有交集的边界框对，并绘制其交集
for i in range(len(boxes1)):
    for j in range(len(boxes2)):
        if iou_matrix1[i, j] > 0:
            tl = torch.max(boxes1[i, :2], boxes2[j, :2])
            br = torch.min(boxes1[i, 2:], boxes2[j, 2:])
            cv2.rectangle(image, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (0, 0, 255), 2)

# 显示图像
cv2.imshow('Bounding Boxes and Intersection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
