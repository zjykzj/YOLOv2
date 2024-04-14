# -*- coding: utf-8 -*-

"""
@Time    : 2024/3/17 20:22
@File    : loss.py
@Author  : zj
@Description:

对于损失函数而言，它应该是支持逐层特征的计算的，不管是YOLOv2Loss还是YOLOv3Loss

"""
from typing import List

import numpy as np

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from utils.torch_utils import de_parallel


def bboxes_iou(bboxes_a: torch.Tensor, bboxes_b: torch.Tensor, xyxy=True) -> torch.Tensor:
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (Tensor): An tensor whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`torch.float32`.
        bbox_b (Tensor): An tensor similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`torch.float32`.
    Returns:
        Tensor:
        An tensor whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.
    """
    # bboxes_a: [N_a, 4]
    # bboxes_b: [N_b, 4]
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # 计算交集矩形的左上角坐标
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # 计算交集矩形的右下角坐标
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
    else:
        # x_center/y_center -> x_top_left, y_top_left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # x_center/y_center -> x_bottom_right/y_bottom_right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

    # 计算bboxes_a的面积
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    # 计算交集的面积
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en

    # 计算IoU
    return area_i / (area_a[:, None] + area_b - area_i)


def make_deltas(box1: Tensor, box2: Tensor) -> Tensor:
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

    def __init__(self, model, ignore_thresh=0.5,
                 coord_scale=1.0, noobj_scale=1.0, obj_scale=5.0, class_scale=1.0):
        super(YOLOv2Loss, self).__init__()
        device = next(model.parameters()).device  # get model device

        m = de_parallel(model).model[-1]  # Detect() module
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        assert self.nl == 1, "YOLOv2Loss supports single-layer feature loss calculation, starting from YOLOv3Loss, it supports multi-layer feature loss"
        self.anchors = m.anchors
        self.device = device

        self.no = self.nc + 5  # number of outputs per anchor
        self.ignore_thresh = ignore_thresh

        self.noobj_scale = noobj_scale
        self.obj_scale = obj_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale

        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid

    def forward(self, p: List[Tensor], targets: Tensor):
        """
        Perform forward pass of the network.

        Args:
            p (List[Tensor]): Predictions made by the network. Each item is a tensor with size [bs, n_anchors, feat_h, feat_w, (xcycwh, conf, n_classes)].
            targets (Tensor): Ground truth targets. Each item format is [image_id, class_id, xc, yc, box_w, box_h].

        Returns:
            tensor: Loss value computed based on predictions and targets.
        """
        assert len(p) == self.nl, "The number of feature layers and prediction layers should be equal."

        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        for i in range(self.nl):
            box_target, box_mask, box_scale, iou_target, iou_mask, class_target, class_mask = \
                self.build_targets(p[i].detach().clone(), targets, i)

            bs, _, ny, nx = p[i].shape  # x(bs,425,20,20) to x(bs,5,20,20,85)
            x = p[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # x/y/conf compress to [0,1]
            xy_conf = torch.sigmoid(x[..., np.r_[:2, 4:5]])
            # exp()
            wh = torch.exp(x[..., 2:4])

            # --------------------------------------
            # box loss
            # [bi, n_anchors, f_h*f_w, 1] -> [bs*n_anchors*f_h*f_w]
            box_mask = box_mask.reshape(-1)
            # [bi, n_anchors, f_h*f_w, 4] -> [bs*n_anchors*f_h*f_w, 4]
            box_target = box_target.reshape(-1, 4)[box_mask > 0]
            box_pred = torch.cat((xy_conf[..., :2], wh), dim=-1).reshape(-1, 4)[box_mask > 0]

            box_scale = torch.sqrt(box_scale.reshape(-1)[box_mask > 0]).reshape(-1, 1)
            box_loss = F.mse_loss(box_pred * box_scale, box_target, reduction='mean')

            # --------------------------------------
            # iou loss
            # [bi, n_anchors, f_h*f_w, 1] -> [bs*n_anchors*f_h*f_w]
            iou_mask = iou_mask.reshape(-1)

            obj_iou_target = iou_target.reshape(-1)[iou_mask == 2]
            obj_iou_pred = xy_conf[..., 2].reshape(-1)[iou_mask == 2]
            obj_iou_loss = F.mse_loss(obj_iou_pred, obj_iou_target, reduction='mean')

            noobj_iou_target = iou_target.reshape(-1)[iou_mask == 1]
            noobj_iou_pred = xy_conf[..., 2].reshape(-1)[iou_mask == 1]
            noobj_iou_loss = F.mse_loss(noobj_iou_pred, noobj_iou_target, reduction='mean')

            # --------------------------------------
            # class loss
            # [bi, n_anchors, f_h*f_w, 1] -> [bs*n_anchors*f_h*f_w]
            class_mask = class_mask.reshape(-1)
            class_target = class_target.reshape(-1)[class_mask > 0]
            class_pred = x[..., 5:].reshape(-1, self.nc)[class_mask > 0]
            class_loss = F.cross_entropy(class_pred, class_target.long(), reduction='mean')

            # calculate the loss, normalized by batch size.
            lcls += class_loss * self.class_scale
            lbox += box_loss * self.obj_scale
            lobj += obj_iou_loss * self.obj_scale + noobj_iou_loss * self.noobj_scale

        return (lbox + lobj + lcls), torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, x, targets, i=0):
        bs, _, ny, nx = x.shape  # x(bs,425,20,20)
        pred_boxes = self._make_pred(x, i)
        iou_target, iou_mask, box_target, box_mask, box_scale, class_target, class_mask = \
            self._build_mask(bs, nx, ny, i)
        # 逐图像计算
        for bi in range(bs):
            # 第bi幅图像的锚点框个数
            num_obj = torch.sum(targets[..., 0] == bi)
            if num_obj == 0:
                # 对于没有标注框的图像，仅计算负样本IOU损失，也就是这幅图像所有的预测框都是负样本
                continue
            gt_targets = targets[targets[..., 0] == bi]
            # [n_bi, 4]
            gt_boxes = gt_targets[..., 2:6]
            # 放大到网格大小
            gt_boxes[..., 0::2] *= nx
            gt_boxes[..., 1::2] *= ny
            # [n_bi]
            gt_cls_ids = gt_targets[..., 1]

            # 第一步：计算所有预测框和所有标注框两两之间的IOU
            #
            # ([n_anchors*f_h*f_w, 4], [num_obj, 4]) -> [n_anchors*f_h*f_w, num_obj]
            ious = bboxes_iou(pred_boxes[bi].reshape(-1, 4), gt_boxes, xyxy=False)
            ious = ious.reshape(self.na, -1, num_obj)
            # 计算每个网格中每个预测框的最大IoU
            # [n_anchors, f_h*f_w, 1]
            max_iou, _ = torch.max(ious, dim=-1, keepdim=True)
            # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold
            # 对于正样本(iou大于阈值), 不参与计算
            n_pos = torch.nonzero(max_iou.view(-1) > self.ignore_thresh).numel()
            if n_pos > 0:
                iou_mask[bi][max_iou >= self.ignore_thresh] = 0

            # 第二步：计算每个网格上锚点框与标注框的IoU，目的：为每个标注框匹配一个对应的预测框
            #
            # [4, n_anchors, f_h, f_w] -> [n_anchors, f_h, f_w, 4] -> [n_anchors, f_h*f_w, 4]
            all_anchors = torch.cat(
                (self.grid[i][:, bi], self.anchor_grid[i][:, bi]), dim=0
            ).permute(1, 2, 3, 0).reshape(self.na, -1, 4)

            # ([n_anchors*f_h*f_w, 4], [num_obj, 4]) -> [n_anchors*f_h*f_w, num_obj]
            overlaps = bboxes_iou(all_anchors.reshape(-1, 4), gt_boxes, xyxy=False).reshape(self.na, -1, num_obj)
            # 逐个锚点框计算，选择最适合的预测框计算损失
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
                # [n_anchors, f_h*f_w, num_obj] -> [n_anchors]
                overlaps_in_cell = overlaps[:, cell_idx, ni].long()
                # 选择IoU最大的锚点框下标
                argmax_anchor_idx = torch.argmax(overlaps_in_cell)
                target_delta = make_deltas(all_anchors[argmax_anchor_idx, cell_idx].unsqueeze(0),
                                           gt_box.unsqueeze(0)).squeeze(0)

                box_target[bi, argmax_anchor_idx, cell_idx, :] = target_delta
                box_mask[bi, argmax_anchor_idx, cell_idx, :] = 1
                # 考虑到xy和wh在不同尺度下（wh对于精度更敏感）进行训练
                pred_box = pred_boxes[bi, argmax_anchor_idx, cell_idx]
                w_i = pred_box[2] / nx
                h_i = pred_box[3] / ny
                box_scale[bi, argmax_anchor_idx, cell_idx, :] = (2 - w_i * h_i)

                # update iou target and iou mask
                iou_target[bi, argmax_anchor_idx, cell_idx, :] = max_iou[argmax_anchor_idx, cell_idx, :]
                iou_mask[bi, argmax_anchor_idx, cell_idx, :] = 2

                # update cls_target, cls_mask
                # 赋值对应类别下标, 对应掩码设置为1
                class_target[bi, argmax_anchor_idx, cell_idx, :] = gt_class
                class_mask[bi, argmax_anchor_idx, cell_idx, :] = 1

        return box_target, box_mask, box_scale, iou_target, iou_mask, class_target, class_mask

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

        # [2, B, n_anchors, F_H, F_W], [2, B, n_anchors, F_H, F_W]
        return torch.stack([x_shift, y_shift]), torch.stack([w_anchors, h_anchors])

    def _make_pred(self, x, i=0):
        bs, _, ny, nx = x.shape  # x(bs,425,20,20)
        x = x.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        if self.grid[i].shape[-2:] != x.shape[2:4]:
            self.grid[i], self.anchor_grid[i] = self._make_grid(bs, nx, ny, i)

        # b_x = sigmoid(t_x) + c_x
        # b_y = sigmoid(t_y) + c_y
        # b_w = p_w * e^t_w
        # b_h = p_h * e^t_h
        #
        # x/y compress to [0,1]
        # [bs, 5, 20, 20, 2]
        xy = torch.sigmoid(x[..., :2])
        xy[..., 0] += self.grid[i][0]
        xy[..., 1] += self.grid[i][1]
        # exp()
        # [bs, 5, 20, 20, 2]
        wh = torch.exp(x[..., 2:4])
        wh[..., 0] *= self.anchor_grid[i][0]
        wh[..., 1] *= self.anchor_grid[i][1]

        # [bs, n_anchors, f_h, f_w, 4] -> [bs, n_anchors, f_h*f_w, 4]
        pred_boxes = torch.cat((xy, wh), dim=4).reshape(bs, self.na, -1, 4)

        return pred_boxes

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
