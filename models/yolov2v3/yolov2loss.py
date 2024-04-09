# -*- coding: utf-8 -*-

"""
@Time    : 2024/3/17 20:22
@File    : loss.py
@Author  : zj
@Description:

对于损失函数而言，它应该是支持逐层特征的计算的，不管是YOLOv2Loss还是YOLOv3Loss

"""
from typing import List

import copy

import numpy as np

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from utils.torch_utils import de_parallel
from utils.general import xywh2xyxy
from utils.metrics import box_iou

"""
特征层输出：[t_x, t_y, t_w, t_h]
锚点框坐标：[c_x, c_y, p_w, p_h]
预测框坐标：[b_x, b_y, b_w, b_h]
标注框坐标：[g_x, g_y, g_w, g_h]

根据特征层输出和锚点框坐标计算预测框坐标

b_x = sigmoid(t_x) + c_x
b_y = sigmoid(t_y) + c_y
b_w = p_w * e^t_w
b_h = p_h * e^t_h

目标是使得[b_x, b_y, b_w, b_h] == [g_x, g_y, g_w, g_h]，上式等价于

g_x = sigmoid(t_x) + c_x
g_y = sigmoid(t_y) + c_y
g_w = p_w * e^t_w
g_h = p_h * e^t_h

所以计算差值如下：

sigmoid(t_x) = g_x - c_x
sigmoid(t_y) = g_y - c_y
e^t_w = g_w / p_w
e^t_h = g_h / p_h
"""


def make_deltas(box1: Tensor, box2: Tensor) -> Tensor:
    """
    Calculate the delta values σ(t_x), σ(t_y), exp(t_w), exp(t_h) used for transforming box1 to box2
    sigmoid(t_x) = b_x - c_x
    sigmoid(t_y) = b_y - c_y
    e^t_w = b_w / p_w
    e^t_h = b_h / p_h

    Arguments:
    box1 -- tensor of shape (N, 4) first set of boxes (c_x, c_y, w, h)
    box2 -- tensor of shape (N, 4) second set of boxes (c_x, c_y, w, h)

    Returns:
    deltas -- tensor of shape (N, 4) delta values (t_x, t_y, t_w, t_h)
                   used for transforming boxes to reference boxes
    """
    assert len(box1.shape) == len(box2.shape) == 2
    # [N, 4] -> [N]
    t_x = box2[:, 0] - box1[:, 0]
    t_y = box2[:, 1] - box1[:, 1]
    t_w = box2[:, 2] / box1[:, 2]
    t_h = box2[:, 3] / box1[:, 3]

    t_x = t_x.view(-1, 1)
    t_y = t_y.view(-1, 1)
    t_w = t_w.view(-1, 1)
    t_h = t_h.view(-1, 1)

    # σ(t_x), σ(t_y), exp(t_w), exp(t_h)
    deltas = torch.cat([t_x, t_y, t_w, t_h], dim=1)
    return deltas


class YOLOv2Loss(nn.Module):

    def __init__(self, model, num_classes=20, ignore_thresh=0.5,
                 coord_scale=1.0, noobj_scale=1.0, obj_scale=5.0, class_scale=1.0):
        super(YOLOv2Loss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        m = de_parallel(model).model[-1]  # Detect() module
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        assert self.nl == 1, "YOLOv2Loss supports single-layer feature loss calculation, starting from YOLOv3Loss, it supports multi-layer feature loss"
        self.anchors = m.anchors
        self.device = device

        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh

        self.noobj_scale = noobj_scale
        self.obj_scale = obj_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale

        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid

    def forward(self, p, targets):  # predictions, targets
        """
        p(bs, n_anchors, feat_h, feat_w, (xcycwh, conf, n_classes))
        targets(image_id, class_id, xc, yc, box_w, box_h)
        """
        for i in range(self.nl):
            iou_target, iou_mask, box_target, box_mask, box_scale, class_target, class_mask = \
                self.build_targets(p.detach().clone(), targets, i)

            bs, _, ny, nx = p[i].shape  # x(bs,425,20,20)
            # [B, C, H, W] -> [B, num_anchors, 5+num_classes, H, W] -> [B, H, W, num_anchors, 5+num_classes]
            outputs = outputs.reshape(B, self.num_anchors, 5 + self.num_classes, H, W) \
                .permute(0, 3, 4, 1, 2)
            # [B, H, W, num_anchors, 5+num_classes] -> [B, H*W*num_anchors, 5+num_classes]
            outputs = outputs.reshape(B, -1, 5 + self.num_classes)
            # x/y/conf compress to [0,1]
            outputs[..., np.r_[:2, 4:5]] = torch.sigmoid(outputs[..., np.r_[:2, 4:5]])
            # exp()
            outputs[..., 2:4] = torch.exp(outputs[..., 2:4])

    def build_targets(self, x, targets, i=0):
        """
        逐层特征计算targets
        """
        bs, _, ny, nx = x[i].shape  # x(bs,425,20,20)
        if self.grid[i].shape[-2:] != x[i].shape[2:4]:
            self.grid[i], self.anchor_grid[i] = self._make_grid(bs, nx, ny, i)

        # b_x = sigmoid(t_x) + c_x
        # b_y = sigmoid(t_y) + c_y
        # b_w = p_w * e^t_w
        # b_h = p_h * e^t_h
        #
        # x/y compress to [0,1]
        xy = torch.sigmoid(x[i][..., :2])
        xy[..., 0] += self.grid[i][0]
        xy[..., 1] += self.grid[i][1]
        # exp()
        wh = torch.exp(x[i][..., 2:4])
        wh[..., 0] *= self.anchor_grid[i][0]
        wh[..., 1] *= self.anchor_grid[i][1]

        # [bs, n_anchors, f_h, f_w, 4] -> [bs, n_anchors, f_h*f_w, 4]
        pred_boxes = torch.cat((xy, wh), dim=4).reshape(bs, self.na, -1, 4)
        # [xc, yc, w, h] -> [x1, y1, x2, y2]
        pred_boxes_xyxy = xywh2xyxy(pred_boxes)
        iou_target, iou_mask, box_target, box_mask, box_scale, class_target, class_mask = \
            self._build_mask(bs, nx, ny, i)
        # 逐个图像进行计算
        for bi in range(bs):
            # 第bi幅图像的锚点框个数
            num_obj = torch.sum(targets[..., 0] == bi)
            if num_obj == 0:
                # 对于没有标注框的图像，不参与损失计算
                # 这一步是否可优化？没有标注框，那么可以计算负样本置信度损失
                iou_mask[bi, ...] = 0
                continue
            # [n_anchors, f_h, f_w, 4] -> [n_anchors, f_h*f_w, 4]
            all_anchors = torch.cat([self.grid[i][bi], self.anchor_grid[i]][bi]).reshape(self.na, -1, 4)

            gt_targets = targets[targets[..., 0] == bi]
            gt_boxes = gt_targets[..., 2:]
            gt_cls_ids = gt_targets[..., 1]

            # 放大到网格大小
            gt_boxes[..., 0::2] *= nx
            gt_boxes[..., 1::2] *= ny
            # [xc, yc, w, h] -> [x1, y1, x2, y2]
            gt_boxes_xyxy = xywh2xyxy(gt_boxes)

            # 第一步：计算所有预测框和所有标注框两两之间的IOU
            # ([n_anchors*f_h*f_w, 4], [num_obj, 4]) -> [n_anchors*f_h*f_w, num_obj]
            ious = box_iou(pred_boxes_xyxy.reshape(-1, 4), gt_boxes_xyxy)
            ious = ious.reshape(self.na, -1, num_obj)
            # 计算每个网格中每个预测框计算得到的最大IoU
            max_iou, _ = torch.max(ious, dim=-1, keepdim=True)
            # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold
            # 对于正样本(iou大于阈值), 不参与计算
            # [H*W, Num_anchors, 1] -> [H*W*Num_anchors] -> [n_pos]
            n_pos = torch.nonzero(max_iou.view(-1) > self.ignore_thresh).numel()
            if n_pos > 0:
                # IOU超过置信度阈值的预测框不参与损失计算
                iou_mask[bi][max_iou >= self.ignore_thresh] = 0

            # 第二步：计算每个网格上锚点框与标注框的IoU，保证每个标注框拥有一个对应的正样本
            # [n_anchors, f_h*f_w, num_obj]
            overlaps = box_iou(all_anchors.reshape(-1, 4), gt_boxes).reshape(self.na, -1, num_obj)

            # 第三步：逐个锚点框计算，选择最适合的预测框计算损失
            # iterate over all objects
            # 每个标注框选择一个锚点框进行训练
            for ni in range(num_obj):
                # compute the center of each gt box to determine which cell it falls on
                # assign it to a specific anchor by choosing max IoU
                # 首先计算锚点框的中心点位于哪个网格, 然后选择其中IoU最大的锚点框参与训练

                # [4]: [xc, yc, w, h]
                gt_box = gt_boxes[ni]
                # 对应的类别下标
                gt_class = gt_cls_ids[ni]
                # 对应网格下标
                cell_idx_x, cell_idx_y = torch.floor(gt_box[:2]).long()
                # 网格列表下标
                cell_idx = cell_idx_y * nx + cell_idx_x

                # update box_target, box_mask
                # 获取该标注框在对应网格上与所有锚点框的IoU
                # [n_anchors, f_h*f_w num_obj] -> [n_anchors]
                overlaps_in_cell = overlaps[:, cell_idx, ni]
                # 选择IoU最大的锚点框下标
                argmax_anchor_idx = torch.argmax(overlaps_in_cell)

                # [H*W, Num_anchors, 4] -> [4]
                # 获取对应网格中指定锚点框的坐标 [x1, y1, w, h]
                target_delta = make_deltas(all_anchors[argmax_anchor_idx, overlaps_in_cell].unsqueeze(0),
                                           gt_box.unsqueeze(0)).squeeze(0)

                # 计算target和mask
                box_target[bi, argmax_anchor_idx, cell_idx, :] = target_delta
                box_mask[bi, argmax_anchor_idx, cell_idx, :] = 1
                pred_box = pred_boxes[bi, argmax_anchor_idx, cell_idx]
                w_i = pred_box[2] / nx
                h_i = pred_box[3] / ny
                box_scale[bi, cell_idx, argmax_anchor_idx, :] = (2 - w_i * h_i)

                # update cls_target, cls_mask
                # 赋值对应类别下标, 对应掩码设置为1
                class_target[bi, cell_idx, argmax_anchor_idx, :] = gt_class
                class_mask[bi, cell_idx, argmax_anchor_idx, :] = 1

                # update iou target and iou mask
                iou_target[bi, cell_idx, argmax_anchor_idx, :] = max_iou[cell_idx, argmax_anchor_idx, :]
                iou_mask[bi, cell_idx, argmax_anchor_idx, :] = 2

        return iou_target, iou_mask, box_target, box_mask, box_scale, class_target, class_mask

    def _make_grid(self, bs, nx=20, ny=20, i=0):
        d = self.device
        t = self.anchors[i].dtype

        # grid coordinate
        # [F] -> [B, n_anchors, F_H, F_W]
        x_shift = torch.broadcast_to(torch.arange(nx), (bs, self.na, ny, nx)).to(dtype=t, device=d)
        # [F] -> [F, 1] -> [B, n_anchors, F_H, F_W]
        y_shift = torch.broadcast_to(torch.arange(ny).reshape(ny, 1), (bs, self.na, ny, nx)).to(dtype=t, device=d)

        # broadcast anchors to all grids
        # [n_anchors] -> [1, n_anchors, 1, 1] -> [B, n_anchors, F_H, F_W]
        w_anchors = torch.broadcast_to(self.anchors[i][:, 0].reshape(1, self.na, 1, 1),
                                       [bs, self.na, ny, nx]).to(dtype=t, device=d)
        h_anchors = torch.broadcast_to(self.anchors[i][:, 1].reshape(1, self.na, 1, 1),
                                       [bs, self.na, ny, nx]).to(dtype=t, device=d)

        return torch.stack([x_shift, y_shift]), torch.stack([w_anchors, h_anchors])

    def _build_mask(self, bs, nx=20, ny=20, i=0):
        d = self.device
        t = self.anchors[i].dtype

        # [B, n_anchors, F_H*F_W, 1]
        iou_target = torch.zeros((bs, self.na, ny * nx, 1)).to(dtype=t, device=d)
        iou_mask = torch.ones((bs, self.na, ny * nx, 1)).to(dtype=t, device=d)

        # [B, n_anchors, F_H*F_W, 4]
        box_target = torch.zeros((bs, self.na, ny * nx, 4)).to(dtype=t, device=d)
        box_mask = torch.zeros((bs, self.na, ny * nx, 1)).to(dtype=t, device=d)
        box_scale = torch.zeros((bs, self.na, ny * nx, 1)).to(dtype=t, device=d)

        # [B, n_anchors, F_H*F_W, 1]
        class_target = torch.zeros((bs, self.na, ny * nx, 1)).to(dtype=t, device=d)
        class_mask = torch.zeros((bs, self.na, ny * nx, 1)).to(dtype=t, device=d)

        return iou_target, iou_mask, box_target, box_mask, box_scale, class_target, class_mask
