# -*- coding: utf-8 -*-

"""
@date: 2023/4/15 下午4:44
@file: yololoss.py
@author: zj
@description:
假定输入的target标注框坐标格式为[x_c, y_c, w, h]
预测框经过计算后，得到的也是[x_c, y_c, w, h]
"""

import numpy as np

import torch
from torch import nn
from torch import Tensor

import torch.nn.functional as F

from yolo.util.box_utils import xywh2xyxy, bboxes_iou


def make_deltas(box1, box2):
    """
    Calculate the delta values σ(t_x), σ(t_y), exp(t_w), exp(t_h) used for transforming box1 to  box2

    Arguments:
    box1 -- tensor of shape (N, 4) first set of boxes (c_x, c_y, w, h)
    box2 -- tensor of shape (N, 4) second set of boxes (c_x, c_y, w, h)

    Returns:
    deltas -- tensor of shape (N, 4) delta values (t_x, t_y, t_w, t_h)
                   used for transforming boxes to reference boxes
    """
    assert len(box1.shape) == len(box2.shape) == 2
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

    def __init__(self, anchors, num_classes=20, ignore_thresh=0.75,
                 coord_scale=1.0, noobj_scale=1.0, obj_scale=5.0, class_scale=1.0):
        super(YOLOv2Loss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh

        self.noobj_scale = noobj_scale
        self.obj_scale = obj_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale

        self.num_anchors = len(anchors)

    def build_mask(self, B, F_size, dtype, device):
        # [B, H*W, num_anchors, 1]
        iou_target = torch.zeros((B, F_size * F_size, self.num_anchors, 1)).to(dtype=dtype, device=device)
        iou_mask = torch.ones((B, F_size * F_size, self.num_anchors, 1)).to(dtype=dtype, device=device)
        iou_mask *= self.noobj_scale

        # [B, H*W, num_anchors, 4]
        box_target = torch.zeros((B, F_size * F_size, self.num_anchors, 4)).to(dtype=dtype, device=device)
        box_mask = torch.zeros((B, F_size * F_size, self.num_anchors, 1)).to(dtype=dtype, device=device)

        # [B, H*W, num_anchors, 1]
        class_target = torch.zeros((B, F_size * F_size, self.num_anchors, 1)).to(dtype=dtype, device=device)
        class_mask = torch.zeros((B, F_size * F_size, self.num_anchors, 1)).to(dtype=dtype, device=device)

        return iou_target, iou_mask, box_target, box_mask, class_target, class_mask

    def build_anchors(self, F_size):
        # grid coordinate
        x_shift = torch.broadcast_to(torch.arange(F_size), (self.num_anchors, F_size, F_size)) \
            # [F_size] -> [F_size, 1] -> [num_anchors, F_size, F_size]
        y_shift = torch.broadcast_to(torch.arange(F_size).reshape(F_size, 1), (self.num_anchors, F_size, F_size))

        anchors = self.anchors * F_size
        # broadcast anchors to all grids
        # [num_anchors] -> [num_anchors, 1, 1] -> [num_anchors, F_size, F_size]
        w_anchors = torch.broadcast_to(anchors[:, 0].reshape(self.num_anchors, 1, 1),
                                       [self.num_anchors, F_size, F_size])
        h_anchors = torch.broadcast_to(anchors[:, 1].reshape(self.num_anchors, 1, 1),
                                       [self.num_anchors, F_size, F_size])

        # [4, num_anchors, F_size, F_size] -> [F_size, F_size, num_anchors, 4]
        # [x_c, y_c, w, h]
        all_anchors = torch.stack([x_shift + 0.5, y_shift + 0.5, w_anchors, h_anchors]).permute(2, 3, 1, 0)
        # [F_size, F_size, num_anchors, 4] -> [F_size*F_size*num_anchors, 4]
        return all_anchors.reshape(F_size * F_size * self.num_anchors, -1)

    def make_pred_boxes(self, outputs):
        dtype = outputs.dtype
        device = outputs.device

        B, C, F_size, _ = outputs.shape[:4]
        # [B, num_anchors * (5+num_classes), H, W] ->
        # [B, num_anchors, 5+num_classes, H, W] ->
        # [B, num_anchors, H, W, 5+num_classes]
        outputs = outputs.reshape(B, self.num_anchors, 5 + self.num_classes, F_size, F_size) \
            .permute(0, 1, 3, 4, 2)

        # grid coordinate
        # [F_size] -> [B, num_anchors, H, W]
        x_shift = torch.broadcast_to(torch.arange(F_size),
                                     (B, self.num_anchors, F_size, F_size)).to(dtype=dtype, device=device)
        # [F_size] -> [f_size, 1] -> [B, num_anchors, H, W]
        y_shift = torch.broadcast_to(torch.arange(F_size).reshape(F_size, 1),
                                     (B, self.num_anchors, F_size, F_size)).to(dtype=dtype, device=device)

        anchors = self.anchors * F_size
        # broadcast anchors to all grids
        # [num_anchors] -> [1, num_anchors, 1, 1] -> [B, num_anchors, H, W]
        w_anchors = torch.broadcast_to(anchors[:, 0].reshape(1, self.num_anchors, 1, 1),
                                       [B, self.num_anchors, F_size, F_size]).to(dtype=dtype, device=device)
        h_anchors = torch.broadcast_to(anchors[:, 1].reshape(1, self.num_anchors, 1, 1),
                                       [B, self.num_anchors, F_size, F_size]).to(dtype=dtype, device=device)

        # b_x = sigmoid(t_x) + c_x
        # b_y = sigmoid(t_y) + c_y
        # b_w = p_w * e^t_w
        # b_h = p_h * e^t_h
        #
        # [B, num_anchors, H, W, 4]
        pred_boxes = outputs[..., :4]
        # x/y/conf compress to [0,1]
        pred_boxes[..., :2] = torch.sigmoid(pred_boxes[..., :2])
        pred_boxes[..., 0] += x_shift
        pred_boxes[..., 1] += y_shift
        # exp()
        pred_boxes[..., 2:4] = torch.exp(pred_boxes[..., 2:4])
        pred_boxes[..., 2] *= w_anchors
        pred_boxes[..., 3] *= h_anchors

        # [B, num_anchors, H, W, 4] -> [B, H, W, num_anchors, 4] -> [B, H*W, num_anchors, 4]
        return pred_boxes.permute(0, 2, 3, 1, 4).reshape(B, F_size * F_size, self.num_anchors, -1)

    def build_targets(self, outputs: Tensor, targets: Tensor):
        B, C, H, W = outputs.shape[:4]
        assert C == self.num_anchors * (5 + self.num_classes)
        assert H == W
        F_size = H

        dtype = outputs.dtype
        device = outputs.device

        # [B, H*W, num_anchors, 4]
        # pred_box: [x_c, y_c, w, h]
        all_pred_boxes = self.make_pred_boxes(outputs)

        # [H*W*num_anchors, 4]
        # [4] = [x_c, y_c, w, h] 坐标相对于网格大小
        all_anchors = self.build_anchors(F_size).to(dtype=dtype, device=device)

        # [B, num_max_det, 5] -> [B, num_max_det] -> [B]
        gt_num_objs = (targets.sum(dim=2) > 0).sum(dim=1)

        iou_target, iou_mask, box_target, box_mask, class_target, class_mask = self.build_mask(B, F_size, dtype, device)
        # 逐图像操作
        for bi in range(B):
            num_obj = gt_num_objs[bi]
            # [num_obj, 4]
            # [4]: [x_c, y_c, w, h]
            gt_boxes = targets[bi][:num_obj][..., :4]
            # [num_obj]
            gt_cls_ids = targets[bi][:num_obj][..., 4]

            # 放大到网格大小
            gt_boxes[..., 0::2] *= F_size
            gt_boxes[..., 1::2] *= F_size
            # [xc, yc, w, h] -> [x1, y1, x2, y2]
            gt_boxes_xxyy = xywh2xyxy(gt_boxes, is_center=True)

            # [H*W, num_anchors, 4] -> [H*W*num_anchors, 4]
            # pred_box: [x_c, y_c, w, h]
            pred_boxes = all_pred_boxes[bi][..., :4].reshape(-1, 4)

            # 首先计算预测框与标注框的IoU，忽略正样本的置信度损失计算
            # ious: [H*W*num_anchors, num_obj]
            ious = bboxes_iou(pred_boxes, gt_boxes, xyxy=False)
            # [H*W*num_anchors, num_obj] -> [H*W, num_anchors, num_obj]
            ious = ious.reshape(-1, self.num_anchors, num_obj)
            # 计算每个网格中每个预测框计算得到的最大IoU
            # shape: (H * W, num_anchors, 1)
            max_iou, _ = torch.max(ious, dim=-1, keepdim=True)

            # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold
            # 对于正样本(iou大于阈值), 不参与计算
            # [H*W, Num_anchors, 1] -> [H*W*Num_anchors] -> [n_pos]
            n_pos = torch.nonzero(max_iou.view(-1) > self.ignore_thresh).numel()
            if n_pos > 0:
                # 如果存在, 那么不参与损失计算
                iou_mask[bi][max_iou >= self.ignore_thresh] = 0

            # 然后计算锚点框与标注框的IoU，保证每个标注框拥有一个对应的正样本
            # overlaps: [H*W*num_anchors, num_obj] -> [H*W, num_anchors, num_obj]
            overlaps = bboxes_iou(all_anchors, gt_boxes, xyxy=False).reshape(-1, self.num_anchors, num_obj)

            # iterate over all objects
            # 每个标注框选择一个锚点框进行训练
            for ni in range(num_obj):
                # compute the center of each gt box to determine which cell it falls on
                # assign it to a specific anchor by choosing max IoU
                # 首先计算锚点框的中心点位于哪个网格, 然后选择其中IoU最大的锚点框参与训练

                # 第t个锚点框 [4]
                # [xc, yc, w, h]
                gt_box = gt_boxes[ni]
                # [x1, y1, x2, y2]
                gt_box_xxyy = gt_boxes_xxyy[ni]
                # 对应的类别下标
                gt_class = gt_cls_ids[ni]
                # 对应网格下标
                cell_idx_x, cell_idx_y = torch.floor(gt_box_xxyy[:2])
                # 网格列表下标
                cell_idx = cell_idx_y * F_size + cell_idx_x
                cell_idx = cell_idx.long()

                # update box_target, box_mask
                # 获取该标注框在对应网格上与所有锚点框的IoU
                # [H*W, num_anchors, num_obj] -> [num_anchors]
                overlaps_in_cell = overlaps[cell_idx, :, ni]
                # 选择IoU最大的锚点框下标
                argmax_anchor_idx = torch.argmax(overlaps_in_cell)

                # [H*W*Num_anchors, 4] -> [H*W, Num_anchors, 4] -> [4]
                # 获取对应网格中指定锚点框的坐标 [xc, yc, w, h]
                response_anchor = all_anchors.view(-1, self.num_anchors, 4)[cell_idx, argmax_anchor_idx, :]
                target_delta = make_deltas(response_anchor.unsqueeze(0), gt_box.unsqueeze(0))

                box_target[bi, cell_idx, argmax_anchor_idx, :] = target_delta.squeeze(0)
                box_mask[bi, cell_idx, argmax_anchor_idx, :] = 1

                # update cls_target, cls_mask
                # 赋值对应类别下标, 对应掩码设置为1
                class_target[bi, cell_idx, argmax_anchor_idx, :] = gt_class
                class_mask[bi, cell_idx, argmax_anchor_idx, :] = 1

                # update iou target and iou mask
                iou_target[bi, cell_idx, argmax_anchor_idx, :] = max_iou[cell_idx, argmax_anchor_idx, :]
                iou_mask[bi, cell_idx, argmax_anchor_idx, :] = self.obj_scale

        # [B, H*W, num_anchors, 1] -> [B, H*W*num_anchors, 1]
        iou_target = iou_target.reshape(B, -1, 1)
        iou_mask = iou_mask.reshape(B, -1, 1)
        # [B, H*W, num_anchors, 4] -> [B, H*W*num_anchors, 4]
        box_target = box_target.reshape(B, -1, 4)
        box_mask = box_mask.reshape(B, -1, 1)
        class_target = class_target.reshape(B, -1, 1).long()
        class_mask = class_mask.reshape(B, -1, 1)

        return iou_target, iou_mask, box_target, box_mask, class_target, class_mask

    def forward(self, outputs, targets):
        iou_target, iou_mask, box_target, box_mask, class_target, class_mask = self.build_targets(outputs, targets)

        B, _, F_size, _ = outputs.shape[:4]
        # [B, C, H, W] -> [B, num_anchors, 5+num_classes, H, W] -> [B, H, W, num_anchors, 5+num_classes]
        outputs = outputs.reshape(B, self.num_anchors, 5 + self.num_classes, F_size, F_size) \
            .permute(0, 3, 4, 1, 2)
        # [B, H, W, num_anchors, 5+num_classes] -> [B, H*W*num_anchors, 5+num_classes]
        outputs = outputs.reshape(B, -1, 5 + self.num_classes)
        # x/y/conf compress to [0,1]
        outputs[..., np.r_[:2, 4:5]] = torch.sigmoid(outputs[..., np.r_[:2, 4:5]])
        # exp()
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4])
        # 分类概率压缩
        outputs[..., 5:] = torch.softmax(outputs[..., 5:], dim=-1)

        # [B, H*W*num_anchors, 5+num_classes] -> [B, H*W*num_anchors, 4]
        pred_deltas = outputs[..., :4]
        # [B, H*W*num_anchors, 5+num_classes] -> [B, H*W*num_anchors, 1]
        pred_confs = outputs[..., 4:5]
        # [B, H*W*num_anchors, 5+num_classes] -> [B, H*W*num_anchors, num_classes]
        pred_probs = outputs[..., 5:]

        # [B, H*W*num_anchors, num_classes] -> [B*H*W*num_anchors, num_classes]
        pred_probs = pred_probs.view(-1, self.num_classes)
        # [B, H*W*num_anchors, 1] -> [B * H * W * num_anchors]
        class_target = class_target.view(-1)
        # [B, H * W * num_anchors, 1] -> [B * H * W * num_anchors]
        class_mask = class_mask.view(-1)

        # ignore the gradient of noobject's target
        class_keep = class_mask.nonzero().squeeze(1)
        # [B*H*W*num_anchors, num_classes] -> [class_keep, num_classes]
        pred_probs = pred_probs[class_keep, :]
        # [B * H * W * num_anchors] -> [class_keep]
        class_target = class_target[class_keep]

        # calculate the loss, normalized by batch size.
        box_loss = F.mse_loss(pred_deltas * box_mask, box_target * box_mask, reduction='sum')
        iou_loss = F.mse_loss(pred_confs * iou_mask, iou_target * iou_mask, reduction='sum')
        class_loss = F.cross_entropy(pred_probs, class_target, reduction='sum')

        loss = (box_loss * self.coord_scale + iou_loss + class_loss * self.class_scale) / B
        return loss
