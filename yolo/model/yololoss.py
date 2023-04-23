# -*- coding: utf-8 -*-

"""
@date: 2023/4/15 下午4:44
@file: yololoss.py
@author: zj
@description: 
"""

import torch

from torch import Tensor
from torch import nn

import torch.nn.functional as F


def generate_all_anchors(anchors, F_H, F_W):
    """
    生成锚点框列表, 每个网格5个锚点框:
    [H, W] -> [H, W, Num_anchors, 4] -> [H*W*Num_anchors, 4]
    Generate dense anchors given grid defined by (H,W)

    Arguments:
    anchors -- tensor of shape (num_anchors, 2), pre-defined anchors (pw, ph) on each cell
    H -- int, grid height
    W -- int, grid width

    Returns:
    all_anchors -- tensor of shape (H * W * num_anchors, 4) dense grid anchors (c_x, c_y, w, h)
    """
    # number of anchors per cell
    # 锚点框个数, YOLOv2使用5个锚点框
    A = anchors.size(0)

    # number of cells
    # 网格数目
    K = F_H * F_W
    # 网格左上角坐标
    # shift_x: [W, H] x表示行 shift_x[0] = [0, 0, 0, ....]
    # shift_y: [W, H] y表示列 shift_y[0] = [0, 1, 2, ...., H]
    shift_x, shift_y = torch.meshgrid([torch.arange(0, F_W), torch.arange(0, F_H)])

    # transpose shift_x and shift_y because we want our anchors to be organized in H x W order
    # [W, H] -> [H, W] x表示列 shift_x[0] = [0, 1, 2, ..., W]
    shift_x = shift_x.t().contiguous()
    # [W, H] -> [H, W] y表示行 shift_y[0] = [0, 0, 0, ...]
    shift_y = shift_y.t().contiguous()

    # shift_x is a long tensor, c_x is a float tensor
    c_x = shift_x.float()
    c_y = shift_y.float()

    # 每个锚点框的左上角坐标位于网格左上角
    # c_x: [H, W] -> [H*W, 1]
    # c_y: [H, W] -> [H*W, 1]
    # cells: [H*W, 2]
    cells = torch.cat([c_x.view(-1, 1), c_y.view(-1, 1)], dim=-1)  # tensor of shape (h * w, 2), (cx, cy)

    # add anchors width and height to centers
    # grid_top_lefts: [H*W, 2] -> [H*W, 1, 2] -> [H*W, A, 2]
    # anchors: [A, 2] -> [1, A, 2] -> [H*W, A, 2]
    # all_anchors: [H*W, A, 4]
    all_anchors = torch.cat([cells.view(K, 1, 2).expand(K, A, 2),
                             anchors.view(1, A, 2).expand(K, A, 2)], dim=-1)

    # [H*W, A, 4] -> [H*W*A, 4]
    all_anchors = all_anchors.view(-1, 4)

    return all_anchors


def xywh2xxyy(box):
    """
    Convert the box encoding format form (c_x, c_y, w, h) to (x1, y1, x2, y2)

    Arguments:
    box -- tensor of shape (N, 4), box of (c_x, c_y, w, h) format

    Returns:
    xxyy_box -- tensor of shape (N, 4), box of (x1, y1, x2, y2) format
    """

    x1 = box[:, 0] - (box[:, 2]) / 2
    y1 = box[:, 1] - (box[:, 3]) / 2
    x2 = box[:, 0] + (box[:, 2]) / 2
    y2 = box[:, 1] + (box[:, 3]) / 2

    x1 = x1.view(-1, 1)
    y1 = y1.view(-1, 1)
    x2 = x2.view(-1, 1)
    y2 = y2.view(-1, 1)

    xxyy_box = torch.cat([x1, y1, x2, y2], dim=1)
    return xxyy_box


def box_transform_inv(box, deltas):
    """
    apply deltas to box to generate predicted boxes

    b_x = σ(t_x) + c_x
    b_y = σ(t_y) + c_y
    b_w = p_w * e^t_w
    b_H = p_h * e^t_h

    Arguments:
    box -- tensor of shape (N, 4), boxes, (c_x, c_y, w, h)
    deltas -- tensor of sh ape (N, 4), deltas, (σ(t_x), σ(t_y), exp(t_w), exp(t_h))

    Returns:
    pred_box -- tensor of shape (N, 4), predicted boxes, (c_x, c_y, w, h)
    """
    # [H*W*Num_anchors] + [H*W*Num_anchors]
    c_x = box[:, 0] + deltas[:, 0]
    c_y = box[:, 1] + deltas[:, 1]
    w = box[:, 2] * deltas[:, 2]
    h = box[:, 3] * deltas[:, 3]

    # [H*W*Num_anchors] -> [H*W*Num_anchors, 1]
    c_x = c_x.view(-1, 1)
    c_y = c_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    pred_box = torch.cat([c_x, c_y, w, h], dim=-1)
    return pred_box


def box_ious(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2 (x1, y1, x2, y2)

    Arguments:
    box1 -- tensor of shape (N, 4), first set of boxes
    box2 -- tensor of shape (K, 4), second set of boxes

    Returns:
    ious -- tensor of shape (N, K), ious between boxes
    """

    N = box1.size(0)
    K = box2.size(0)

    # when torch.max() takes tensor of different shape as arguments, it will broadcasting them.
    xi1 = torch.max(box1[:, 0].view(N, 1), box2[:, 0].view(1, K))
    yi1 = torch.max(box1[:, 1].view(N, 1), box2[:, 1].view(1, K))
    xi2 = torch.min(box1[:, 2].view(N, 1), box2[:, 2].view(1, K))
    yi2 = torch.min(box1[:, 3].view(N, 1), box2[:, 3].view(1, K))

    # we want to compare the compare the value with 0 elementwise. However, we can't
    # simply feed int 0, because it will invoke the function torch(max, dim=int) which is not
    # what we want.
    # To feed a tensor 0 of same type and device with box1 and box2
    # we use tensor.new().fill_(0)

    iw = torch.max(xi2 - xi1, box1.new(1).fill_(0))
    ih = torch.max(yi2 - yi1, box1.new(1).fill_(0))

    inter = iw * ih

    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    box1_area = box1_area.view(N, 1)
    box2_area = box2_area.view(1, K)

    union_area = box1_area + box2_area - inter

    ious = inter / union_area

    return ious


def xxyy2xywh(box):
    """
    Convert the box (x1, y1, x2, y2) encoding format to (c_x, c_y, w, h) format

    Arguments:
    box: tensor of shape (N, 4), boxes of (x1, y1, x2, y2) format

    Returns:
    xywh_box: tensor of shape (N, 4), boxes of (c_x, c_y, w, h) format
    """

    c_x = (box[:, 2] + box[:, 0]) / 2
    c_y = (box[:, 3] + box[:, 1]) / 2
    w = box[:, 2] - box[:, 0]
    h = box[:, 3] - box[:, 1]

    c_x = c_x.view(-1, 1)
    c_y = c_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    xywh_box = torch.cat([c_x, c_y, w, h], dim=1)
    return xywh_box


def box_transform(box1, box2):
    """
    Calculate the delta values σ(t_x), σ(t_y), exp(t_w), exp(t_h) used for transforming box1 to  box2

    Arguments:
    box1 -- tensor of shape (N, 4) first set of boxes (c_x, c_y, w, h)
    box2 -- tensor of shape (N, 4) second set of boxes (c_x, c_y, w, h)

    Returns:
    deltas -- tensor of shape (N, 4) delta values (t_x, t_y, t_w, t_h)
                   used for transforming boxes to reference boxes
    """

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

    def build_target(self, pred_box_deltas, pred_confs, targets, F_H, F_W):
        """
        delta_pred: [B, F_H*F_W*num_anchors, 4]
        targets: [B, num_max_det, 5]
        """
        # [B] 批量大小
        B = pred_box_deltas.shape[0]

        # [B, num_max_det,  4]
        gt_boxes_batch = targets[..., :4]
        # [B, num_max_det]
        gt_classes_batch = targets[..., 4]
        # [B, num_max_det, 5] -> [B, num_max_det, 4] -> [B, num_max_det] -> [B]
        gt_num_objs = (gt_boxes_batch.sum(dim=2) > 0).sum(dim=1)

        # 接下来就是创建各种target和mask
        #
        # iou_target: [N, H*W*Num_anchors, 1]
        iou_target = pred_box_deltas.new_zeros((B, F_H * F_W, self.num_anchors, 1))
        # iou_mask: [N, H*W, Num_anchors, 1]
        # 置信度掩码, 包含了负责标注框的置信度预测(趋近于1)以及不包含目标的网格以及锚点框的置信度预测(趋近于0)
        # 默认情况下, 所以预测框对应的预测置信度都参与计算. 所以mask = 1
        # 又因为大部分都是不负责标注框的预测, 所以乘以noobj_scale系数
        iou_mask = pred_box_deltas.new_ones((B, F_H * F_W, self.num_anchors, 1)) * self.noobj_scale

        # target和mask, 看起来target用于损失计算的真值, mask用于判断哪些预测框用于损失计算
        # [N, H*W, Num_anchors, 4]
        box_target = pred_box_deltas.new_zeros((B, F_H * F_W, self.num_anchors, 4))
        # [N, H*W, Num_anchors, 1]
        box_mask = pred_box_deltas.new_zeros((B, F_H * F_W, self.num_anchors, 1))

        # [N, H*W, Num_anchors, 1]
        class_target = pred_confs.new_zeros((B, F_H * F_W, self.num_anchors, 1))
        # [N, H*W, Num_anchors, 1]
        class_mask = pred_confs.new_zeros((B, F_H * F_W, self.num_anchors, 1))

        # get all the anchors
        # 生成网格锚点框
        # [H*W*Num_anchors, 4] [x1, y1, box_w, box_h]
        all_anchors_x1y1wh = generate_all_anchors(self.anchors, F_H, F_W)
        # [H*W*Num_anchors, 4] -> [N, H*W*Num_anchors, 4]
        # 这一步的目的是为了保持相同的数据类型torch.dtype以及所处设备torch.device
        all_anchors_x1y1wh = pred_box_deltas.new(*all_anchors_x1y1wh.size()).copy_(all_anchors_x1y1wh)

        all_anchors_xcycwh = all_anchors_x1y1wh.clone()
        # [x1, y1, w, h] -> [xc, yc, w, h]
        all_anchors_xcycwh[:, 0:2] += 0.5
        # [xc, yc, w, h] -> [x1, y1, x2, y2]
        all_anchors_xxyy = xywh2xxyy(all_anchors_xcycwh)

        # process over batches
        # 逐个图像进行操作
        for bi in range(B):
            # 确认每个图像有效的标注框数目
            num_obj = gt_num_objs[bi].item()
            # 获取YOLOv2预测结果
            # [H*W*Num_anchors, 4]
            box_deltas = pred_box_deltas[bi]
            # 获取标注框数据
            # gt_box: [x1, y1, x2, y2]
            # gt_boxes: [num_obj, 4]
            gt_boxes = gt_boxes_batch[bi][:num_obj, :]
            # 获取标注框对应类别下标
            gt_classes = gt_classes_batch[bi][:num_obj]

            # rescale ground truth boxes
            # 缩放到特征图大小
            gt_boxes[:, 0::2] *= F_W
            gt_boxes[:, 1::2] *= F_H

            # step 1: process IoU target
            # apply box_deltas to pre-defined anchors
            #
            # 结合预测结果和锚点框得到预测框
            # [H*W*Num_anchors, 4]
            pred_boxes = box_transform_inv(all_anchors_x1y1wh, box_deltas)
            # [xc, yc, w, h] -> [x1, y1, x2, y2]
            pred_boxes = xywh2xxyy(pred_boxes)

            # for each anchor, its iou target is corresponded to the max iou with any gt boxes
            # 计算预测框和标注框之间的iou, 最大的预测框(并且IoU超过了阈值)负责该标注框的训练
            #
            # [H*W*Num_anchors, 4], [num_obj, 4] -> [H*W*Num_anchors, num_obj]
            ious = box_ious(pred_boxes, gt_boxes)
            # [H*W*Num_anchors, num_obj] -> [H*W, Num_anchors, num_obj]
            ious = ious.view(-1, self.num_anchors, num_obj)
            # 计算每个网格中和标注框最大的iou
            # shape: (H * W, num_anchors, 1)
            max_iou, _ = torch.max(ious, dim=-1, keepdim=True)

            # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold
            # 对于正样本(iou大于阈值), 不参与计算
            # [H*W, Num_anchors, 1] -> [H*W*Num_anchors]
            iou_thresh_filter = max_iou.view(-1) > self.ignore_thresh
            n_pos = torch.nonzero(iou_thresh_filter).numel()

            if n_pos > 0:
                # 如果存在, 那么不参与损失计算
                iou_mask[bi][max_iou >= self.ignore_thresh] = 0

            # step 2: process box target and class target
            # calculate overlaps between anchors and gt boxes
            #
            # 如何确定正样本和负样本???
            # 应该是锚点框和真值框进行匹配, 哪个锚点框负责真值框预测, 预测框的目的是为了让锚点框更好的拟合标注框!!!
            # [H*W*Num_anchors, 4], [num_obj, 4] -> [H*W*Num_anchors, num_obj] -> [H*W, Num_anchors, num_obj]
            overlaps = box_ious(all_anchors_xxyy, gt_boxes).view(-1, self.num_anchors, num_obj)
            # [x1, y1, x2, y2] -> [xc, yc, w, h]
            gt_boxes_xcycwh = xxyy2xywh(gt_boxes)

            # iterate over all objects
            # 每个标注框选择一个锚点框进行训练
            for t in range(len(gt_boxes)):
                # compute the center of each gt box to determine which cell it falls on
                # assign it to a specific anchor by choosing max IoU
                # 首先计算锚点框的中心点位于哪个网格, 然后选择其中IoU最大的锚点框参与训练

                # 第t个锚点框
                gt_box_xcycwh = gt_boxes_xcycwh[t]
                # 对应的类别下标
                gt_class = gt_classes[t]
                # 对应网格下标
                cell_idx_x, cell_idx_y = torch.floor(gt_box_xcycwh[:2])
                # 网格列表下标
                cell_idx = cell_idx_y * F_W + cell_idx_x
                cell_idx = cell_idx.long()

                # update box_target, box_mask
                # 获取该标注框在对应网格上与所有锚点框的IoU
                # [H*W, Num_anchors, num_obj] -> [Num_anchors]
                overlaps_in_cell = overlaps[cell_idx, :, t]
                # 选择IoU最大的锚点框下标
                argmax_anchor_idx = torch.argmax(overlaps_in_cell)

                # [H*W*Num_anchors, 4] -> [H*W, Num_anchors, 4] -> [4] -> [1, 4]
                # 获取对应网格中指定锚点框的坐标 [x1, y1, w, h]
                assigned_grid = \
                    all_anchors_x1y1wh.view(-1, self.num_anchors, 4)[cell_idx, argmax_anchor_idx, :].unsqueeze(0)
                # [4] -> [1, 4]
                gt_box = gt_box_xcycwh.unsqueeze(0)
                # 锚点框和标注框之间的差距就是YOLOv2需要学习的偏移
                target_t = box_transform(assigned_grid, gt_box)
                # 赋值, 对应掩码下标设置为1
                box_target[bi, cell_idx, argmax_anchor_idx, :] = target_t.unsqueeze(0)
                box_mask[bi, cell_idx, argmax_anchor_idx, :] = 1

                # update cls_target, cls_mask
                # 同步赋值对应类别下标, 对应掩码设置为1
                class_target[bi, cell_idx, argmax_anchor_idx, :] = gt_class
                class_mask[bi, cell_idx, argmax_anchor_idx, :] = 1

                # update iou target and iou mask
                iou_target[bi, cell_idx, argmax_anchor_idx, :] = max_iou[cell_idx, argmax_anchor_idx, :]
                iou_mask[bi, cell_idx, argmax_anchor_idx, :] = self.obj_scale

        return iou_target.view(B, -1, 1), iou_mask.view(B, -1, 1), \
            box_target.view(B, -1, 4), box_mask.view(B, -1, 1), \
            class_target.view(B, -1, 1).long(), class_mask.view(B, -1, 1)

    def forward(self, outputs, targets):
        """
        先仿照yolov2一比一复制, 然后再进行优化

        结合预测框输出 / 锚点框坐标 和 网格坐标, 计算最终的预测框坐标
        b_x = sigmoid(t_x) + c_x
        b_y = sigmoid(t_y) + c_y
        b_w = p_w * e^t_w
        b_h = p_h * e^t_h
        """
        # out -- tensor of shape (B, num_anchors * (5 + num_classes), H, W)
        B, _, F_H, F_W = outputs.shape[:4]
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W*Num_anchors, 5+Num_classes]
        outputs = outputs.permute(0, 2, 3, 1).contiguous().view(B, F_H * F_W * self.num_anchors, 5 + self.num_classes)

        # activate the output tensor
        # `sigmoid` for t_x, t_y, t_c; `exp` for t_h, t_w;
        # `softmax` for (class1_score, class2_score, ...)
        #
        # [N, H*W*Num_anchors, 2]
        xy_pred = torch.sigmoid(outputs[:, :, 0:2])
        # [N, H*W*Num_anchors, 2]
        hw_pred = torch.exp(outputs[:, :, 2:4])
        # [N, H*W*Num_anchors, 1]
        conf_pred = torch.sigmoid(outputs[:, :, 4:5])

        # [N, H*W*Num_anchors, Num_classes]
        class_score = outputs[:, :, 5:]

        # [N, H*W*Num_anchors, 4]
        box_delta = torch.cat([xy_pred, hw_pred], dim=-1)

        iou_target, iou_mask, box_target, box_mask, class_target, class_mask = \
            self.build_target(box_delta, conf_pred, targets, F_H, F_W)

        b, _, num_classes = class_score.size()
        # [B, H * W * num_anchors, num_classes] -> [B * H * W * num_anchors, num_classes]
        class_score_batch = class_score.view(-1, num_classes)
        # [B, H * W * num_anchors, 1] -> [B * H * W * num_anchors]
        class_target = class_target.view(-1)
        # [B, H * W * num_anchors, 1] -> [B * H * W * num_anchors]
        class_mask = class_mask.view(-1)

        # ignore the gradient of noobject's target
        class_keep = class_mask.nonzero().squeeze(1)
        class_score_batch_keep = class_score_batch[class_keep, :]
        class_target_keep = class_target[class_keep]

        # calculate the loss, normalized by batch size.
        box_loss = self.coord_scale * F.mse_loss(box_delta * box_mask, box_target * box_mask, reduction='sum') / 2.0
        iou_loss = F.mse_loss(conf_pred * iou_mask, iou_target * iou_mask, reduction='sum') / 2.0
        class_loss = self.class_scale * F.cross_entropy(class_score_batch_keep, class_target_keep, reduction='sum')

        total_loss = box_loss + iou_loss + class_loss
        return total_loss / B
