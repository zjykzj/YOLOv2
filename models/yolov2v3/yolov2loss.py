# -*- coding: utf-8 -*-

"""
@date: 2023/4/15 下午4:44
@file: yololoss.py
@author: zj
@description:
假定输入的target标注框坐标格式为[x_c, y_c, w, h]
预测框经过计算后，得到的也是[x_c, y_c, w, h]

锚点框的宽高是相对于整幅图像的，同样的，标注框的宽高也是相对于整幅图像的，所以可以计算锚点框与标注框之间的IoU

同样的，预测框经过转换后也是相对于整幅图像的，所以可以计算预测框与标注框之间的IoU
"""

import numpy as np

import torch
from torch import nn
from torch import Tensor

import torch.nn.functional as F


# from yolo.util.box_utils import xywh2xyxy, bboxes_iou


def bboxes_iou(bboxes_a: Tensor, bboxes_b: Tensor, xyxy=True) -> Tensor:
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    # bboxes_a: [N_a, 4]
    # bboxes_b: [N_b, 4]
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        # xyxy: x_top_left, y_top_left, x_bottom_right, y_bottom_right
        # 计算交集矩形的左上角坐标
        # torch.max([N_a, 1, 2], [N_b, 2]) -> [N_a, N_b, 2]
        # torch.max: 双重循环
        #   第一重循环 for i in range(N_a)，遍历boxes_a, 获取边界框i，大小为[2]
        #       第二重循环　for j in range(N_b)，遍历bboxes_b，获取边界框j，大小为[2]
        #           分别比较i[0]/j[0]和i[1]/j[1]，获取得到最大的x/y
        #   遍历完成后，获取得到[N_a, N_b, 2]
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        # 计算交集矩形的右下角坐标
        # torch.min([N_a, 1, 2], [N_b, 2]) -> [N_a, N_b, 2]
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # 计算bboxes_a的面积
        # x_bottom_right/y_bottom_right - x_top_left/y_top_left = w/h
        # prod([N, w/h], 1) = [N], 每个item表示边界框的面积w*h
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # x_center/y_center -> x_top_left, y_top_left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        # x_center/y_center -> x_bottom_right/y_bottom_right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # prod([N_a, w/h], 1) = [N_a], 每个item表示边界框的面积w*h
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    # 判断符合条件的结果：x_top_left/y_top_left < x_bottom_right/y_bottom_right
    # [N_a, N_b, 2] < [N_a, N_b, 2] = [N_a, N_b, 2]
    # prod([N_a, N_b, 2], 2) = [N_a, N_b], 数值为1/0
    en = (tl < br).type(tl.type()).prod(dim=2)
    # 首先计算交集w/h: [N_a, N_b, 2] - [N_a, N_b, 2] = [N_a, N_b, 2]
    # 然后计算交集面积：prod([N_a, N_b, 2], 2) = [N_a, N_b]
    # 然后去除不符合条件的交集面积
    # [N_a, N_b] * [N_a, N_b](数值为1/0) = [N_a, N_b]
    # 大小为[N_a, N_b]，表示bboxes_a的每个边界框与bboxes_b的每个边界框之间的IoU
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())

    # 计算IoU
    # 首先计算所有面积
    # area_a[:, None] + area_b - area_i =
    # [N_a, 1] + [N_b] - [N_a, N_b] = [N_a, N_b]
    # 然后交集面积除以所有面积，计算IoU
    # [N_a, N_b] / [N_a, N_b] = [N_a, N_b]
    return area_i / (area_a[:, None] + area_b - area_i)


def xywh2xyxy(boxes, is_center=False):
    assert len(boxes.shape) >= 2 and boxes.shape[-1] == 4
    if isinstance(boxes, np.ndarray):
        boxes_xxyy = np.zeros(boxes.shape)
    else:
        boxes_xxyy = boxes.new_zeros(boxes.shape)
    if is_center:
        # [x_c, y_c, w, h] -> [x1, y1, x2, y2]
        boxes_xxyy[..., 0] = (boxes[..., 0] - boxes[..., 2] / 2)
        boxes_xxyy[..., 1] = (boxes[..., 1] - boxes[..., 3] / 2)
        boxes_xxyy[..., 2] = (boxes[..., 0] + boxes[..., 2] / 2)
        boxes_xxyy[..., 3] = (boxes[..., 1] + boxes[..., 3] / 2)
    else:
        # [x1, y1, w, h] -> [x1, y1, x2, y2]
        boxes_xxyy[..., 0] = boxes[..., 0]
        boxes_xxyy[..., 1] = boxes[..., 1]
        boxes_xxyy[..., 2] = (boxes[..., 0] + boxes[..., 2])
        boxes_xxyy[..., 3] = (boxes[..., 1] + boxes[..., 3])
    return boxes_xxyy


def make_deltas(box1: Tensor, box2: Tensor) -> Tensor:
    """
    Calculate the delta values σ(t_x), σ(t_y), exp(t_w), exp(t_h) used for transforming box1 to  box2
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

        from utils.torch_utils import de_parallel
        m = de_parallel(model).model[-1]  # Detect() module
        # self.na = m.na  # number of anchors
        # self.nc = m.nc  # number of classes
        # self.nl = m.nl  # number of layers
        # assert self.nl == 1, "YOLOv2Loss supports single-layer feature loss calculation, starting from YOLOv3Loss, it supports multi-layer feature loss"
        self.anchors = m.anchors[0]
        self.num_anchors = len(self.anchors)
        print(f"m.anchors: {self.anchors} - shape: {self.anchors.shape}")
        self.device = device
        self.num_classes = m.nc

        # self.no = self.nc + 5  # number of outputs per anchor
        self.ignore_thresh = ignore_thresh

        self.noobj_scale = noobj_scale
        self.obj_scale = obj_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale

        # self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        # self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid

    # def __init__(self, anchors, num_classes=20, ignore_thresh=0.5,
    #              coord_scale=1.0, noobj_scale=1.0, obj_scale=1.0, class_scale=1.0):
    #     super(YOLOv2Loss, self).__init__()
    #     self.anchors = anchors
    #     self.num_classes = num_classes
    #     self.ignore_thresh = ignore_thresh
    #
    #     self.noobj_scale = noobj_scale
    #     self.obj_scale = obj_scale
    #     self.class_scale = class_scale
    #     self.coord_scale = coord_scale
    #
    #     self.num_anchors = len(anchors)

    def build_mask(self, B, H, W, dtype, device):
        # [B, H*W, num_anchors, 1]
        iou_target = torch.zeros((B, H * W, self.num_anchors, 1)).to(dtype=dtype, device=device)
        iou_mask = torch.ones((B, H * W, self.num_anchors, 1)).to(dtype=dtype, device=device)

        # [B, H*W, num_anchors, 4]
        box_target = torch.zeros((B, H * W, self.num_anchors, 4)).to(dtype=dtype, device=device)
        box_mask = torch.zeros((B, H * W, self.num_anchors, 1)).to(dtype=dtype, device=device)
        box_scale = torch.zeros((B, H * W, self.num_anchors, 1)).to(dtype=dtype, device=device)

        # [B, H*W, num_anchors, 1]
        class_target = torch.zeros((B, H * W, self.num_anchors, 1)).to(dtype=dtype, device=device)
        class_mask = torch.zeros((B, H * W, self.num_anchors, 1)).to(dtype=dtype, device=device)

        return iou_target, iou_mask, box_target, box_mask, box_scale, class_target, class_mask

    def make_pred_boxes(self, outputs):
        dtype = outputs.dtype
        device = outputs.device

        B, C, H, W = outputs.shape[:4]
        # [B, num_anchors * (5+num_classes), H, W] ->
        # [B, num_anchors, 5+num_classes, H, W] ->
        # [B, num_anchors, H, W, 5+num_classes]
        outputs = outputs.reshape(B, self.num_anchors, 5 + self.num_classes, H, W) \
            .permute(0, 1, 3, 4, 2)

        # grid coordinate
        # [F_size] -> [num_anchors, H, W]
        x_shift = torch.broadcast_to(torch.arange(W),
                                     (self.num_anchors, H, W)).to(dtype=dtype, device=device)
        # [F_size] -> [f_size, 1] -> [num_anchors, H, W]
        y_shift = torch.broadcast_to(torch.arange(H).reshape(H, 1),
                                     (self.num_anchors, H, W)).to(dtype=dtype, device=device)

        # broadcast anchors to all grids
        # [num_anchors] -> [num_anchors, 1, 1] -> [num_anchors, H, W]
        w_anchors = torch.broadcast_to(self.anchors[:, 0].reshape(self.num_anchors, 1, 1),
                                       [self.num_anchors, H, W]).to(dtype=dtype, device=device)
        h_anchors = torch.broadcast_to(self.anchors[:, 1].reshape(self.num_anchors, 1, 1),
                                       [self.num_anchors, H, W]).to(dtype=dtype, device=device)

        # b_x = sigmoid(t_x) + c_x
        # b_y = sigmoid(t_y) + c_y
        # b_w = p_w * e^t_w
        # b_h = p_h * e^t_h
        #
        # [B, num_anchors, H, W, 4]
        pred_boxes = outputs[..., :4]
        # x/y compress to [0,1]
        pred_boxes[..., :2] = torch.sigmoid(pred_boxes[..., :2])
        pred_boxes[..., 0] += x_shift.expand(B, self.num_anchors, H, W)
        pred_boxes[..., 1] += y_shift.expand(B, self.num_anchors, H, W)
        # exp()
        pred_boxes[..., 2:4] = torch.exp(pred_boxes[..., 2:4])
        pred_boxes[..., 2] *= w_anchors.expand(B, self.num_anchors, H, W)
        pred_boxes[..., 3] *= h_anchors.expand(B, self.num_anchors, H, W)

        # [B, num_anchors, H, W, 4] -> [B, H, W, num_anchors, 4] -> [B, H*W, num_anchors, 4]
        pred_boxes = pred_boxes.permute(0, 2, 3, 1, 4).reshape(B, H * W, self.num_anchors, 4)
        # print(torch.max(pred_boxes))
        pred_boxes[..., 0::2] = torch.clamp(pred_boxes[..., 0::2], 0, W)
        pred_boxes[..., 1::2] = torch.clamp(pred_boxes[..., 1::2], 0, H)

        # [4, num_anchors, H, W] -> [H, W, num_anchors, 4]
        # [x_c, y_c, w, h]
        all_anchors_x1y1 = torch.stack([x_shift, y_shift, w_anchors, h_anchors]).permute(2, 3, 1, 0)
        # [H, W, num_anchors, 4] -> [H*W, num_anchors, 4]
        all_anchors_x1y1 = all_anchors_x1y1.reshape(H * W, self.num_anchors, -1)

        return pred_boxes, all_anchors_x1y1

    def build_targets(self, outputs: Tensor, targets: Tensor):
        B, C, H, W = outputs.shape[:4]
        assert C == self.num_anchors * (5 + self.num_classes)

        dtype = outputs.dtype
        device = outputs.device

        # all_pred_boxes: [B, H*W, num_anchors, 4]
        # all_anchors_xcyc: [H*W, num_anchors, 4]
        # [4] = [x_c, y_c, w, h] 坐标相对于网格大小
        all_pred_boxes, all_anchors_x1y1 = self.make_pred_boxes(outputs)
        all_anchors_xcyc = all_anchors_x1y1.clone()
        all_anchors_xcyc[..., :2] += 0.5

        # [B, num_max_det, 5] -> [B, num_max_det] -> [B]
        gt_num_objs = (targets.sum(dim=2) > 0).sum(dim=1)

        iou_target, iou_mask, box_target, box_mask, box_scale, class_target, class_mask = \
            self.build_mask(B, H, W, dtype, device)
        # 逐图像操作
        for bi in range(B):
            num_obj = gt_num_objs[bi]
            if num_obj == 0:
                # 对于没有标注框的图像，不参与损失计算
                iou_mask[bi, ...] = 0
                continue
            # [num_obj, 4]
            # [4]: [x_c, y_c, w, h]
            gt_boxes = targets[bi][:num_obj][..., 1:]
            # [num_obj]
            gt_cls_ids = targets[bi][:num_obj][..., 0]

            # 放大到网格大小
            gt_boxes[..., 0::2] *= W
            gt_boxes[..., 1::2] *= H
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
            overlaps = bboxes_iou(all_anchors_xcyc.reshape(-1, 4),
                                  gt_boxes, xyxy=False).reshape(-1, self.num_anchors, num_obj)

            # iterate over all objects
            # 每个标注框选择一个锚点框进行训练
            for ni in range(num_obj):
                # compute the center of each gt box to determine which cell it falls on
                # assign it to a specific anchor by choosing max IoU
                # 首先计算锚点框的中心点位于哪个网格, 然后选择其中IoU最大的锚点框参与训练

                # 第t个标注框
                # [4]: [xc, yc, w, h]
                gt_box = gt_boxes[ni]
                # [x1, y1, x2, y2]
                gt_box_xxyy = gt_boxes_xxyy[ni]
                # 对应的类别下标
                gt_class = gt_cls_ids[ni]
                # 对应网格下标
                cell_idx_x, cell_idx_y = torch.floor(gt_box[:2])
                # 网格列表下标
                cell_idx = cell_idx_y * W + cell_idx_x
                cell_idx = cell_idx.long()

                # update box_target, box_mask
                # 获取该标注框在对应网格上与所有锚点框的IoU
                # [H*W, num_anchors, num_obj] -> [num_anchors]
                overlaps_in_cell = overlaps[cell_idx, :, ni]
                # 选择IoU最大的锚点框下标
                argmax_anchor_idx = torch.argmax(overlaps_in_cell)

                # [H*W, Num_anchors, 4] -> [4]
                # 获取对应网格中指定锚点框的坐标 [x1, y1, w, h]
                response_anchor_x1y1 = all_anchors_x1y1[cell_idx, argmax_anchor_idx, :]
                target_delta = make_deltas(response_anchor_x1y1.unsqueeze(0), gt_box.unsqueeze(0)).squeeze(0)

                box_target[bi, cell_idx, argmax_anchor_idx, :] = target_delta
                box_mask[bi, cell_idx, argmax_anchor_idx, :] = 1
                pred_box = all_pred_boxes[bi, cell_idx, argmax_anchor_idx]
                w_i = pred_box[2] / W
                h_i = pred_box[3] / H
                # if torch.isnan(torch.sqrt((2 - w_i * h_i))):
                #     print(f"w_i: {w_i} - h_i: {h_i} - pred_box: {pred_box}")
                #     exit(1)
                # assert 0 <= w_i <= 1 and 0 <= h_i <= 1, f"w_i: {w_i} - h_i: {h_i} - pred_box: {pred_box}"
                box_scale[bi, cell_idx, argmax_anchor_idx, :] = (2 - w_i * h_i)

                # update cls_target, cls_mask
                # 赋值对应类别下标, 对应掩码设置为1
                class_target[bi, cell_idx, argmax_anchor_idx, :] = gt_class
                class_mask[bi, cell_idx, argmax_anchor_idx, :] = 1

                # update iou target and iou mask
                iou_target[bi, cell_idx, argmax_anchor_idx, :] = max_iou[cell_idx, argmax_anchor_idx, :]
                iou_mask[bi, cell_idx, argmax_anchor_idx, :] = 2

        # [B, H*W, num_anchors, 1] -> [B*H*W*num_anchors]
        iou_target = iou_target.reshape(-1)
        iou_mask = iou_mask.reshape(-1)
        # [B, H*W, num_anchors, 4] -> [B*H*W*num_anchors, 4]
        box_target = box_target.reshape(-1, 4)
        box_mask = box_mask.reshape(-1)
        box_scale = box_scale.reshape(-1, 1)
        class_target = class_target.reshape(-1).long()
        class_mask = class_mask.reshape(-1)

        return iou_target, iou_mask, box_target, box_mask, box_scale, class_target, class_mask

    def forward(self, p, targets):
        """
        计算损失需要得到
        1. 标注框坐标和锚点框坐标之间的delta（作为target）
        2. 输出卷积特征生成的预测框delta（作为预测结果）
        """
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        lcls = torch.zeros(1, device=self.device)  # class loss

        for outputs in p:
            bs, _, ny, nx, _ = outputs.shape
            # x(bs,5,20,20,85) -> x(bs,5,85,20,20) -> x(bs,425,20,20)
            outputs = outputs.permute(0, 1, 4, 2, 3).reshape(bs, -1, ny, nx).contiguous()

            new_targets = torch.zeros(bs, 50, 5).to(self.device)
            for bi in range(bs):
                if torch.sum(targets[..., 0] == bi) > 0:
                    sub_targets = targets[targets[..., 0] == bi]
                    new_targets[bi][:sub_targets.size(0)] = sub_targets[..., 1:]

            iou_target, iou_mask, box_target, box_mask, box_scale, class_target, class_mask = \
                self.build_targets(outputs.detach().clone(), new_targets)

            B, _, H, W = outputs.shape[:4]
            # [B, C, H, W] -> [B, num_anchors, 5+num_classes, H, W] -> [B, H, W, num_anchors, 5+num_classes]
            outputs = outputs.reshape(B, self.num_anchors, 5 + self.num_classes, H, W) \
                .permute(0, 3, 4, 1, 2)
            # [B, H, W, num_anchors, 5+num_classes] -> [B, H*W*num_anchors, 5+num_classes]
            outputs = outputs.reshape(B, -1, 5 + self.num_classes)
            # x/y/conf compress to [0,1]
            outputs[..., np.r_[:2, 4:5]] = torch.sigmoid(outputs[..., np.r_[:2, 4:5]])
            # exp()
            outputs[..., 2:4] = torch.exp(outputs[..., 2:4])

            # [B, H*W*num_anchors, 5+num_classes] -> [B, H*W*num_anchors, 4] -> [B*H*W*num_anchors, 4]
            pred_deltas = outputs[..., :4].reshape(-1, 4)
            # [B, H*W*num_anchors, 5+num_classes] -> [B, H*W*num_anchors, 1] -> [B*H*W*num_anchors]
            pred_confs = outputs[..., 4:5].reshape(-1)
            # [B, H*W*num_anchors, 5+num_classes] -> [B, H*W*num_anchors, num_classes] -> [B*H*W*num_anchors, num_classes]
            pred_probs = outputs[..., 5:].reshape(-1, self.num_classes)

            # --------------------------------------
            # box loss
            pred_deltas = pred_deltas[box_mask > 0]
            box_target = box_target[box_mask > 0]
            box_scale = torch.sqrt(box_scale[box_mask > 0])
            box_loss = F.mse_loss(pred_deltas * box_scale, box_target * box_scale, reduction='sum')
            # print(box_scale)

            # --------------------------------------
            # iou loss
            obj_pred_confs = pred_confs[iou_mask == 2]
            obj_iou_target = iou_target[iou_mask == 2]
            obj_iou_loss = F.mse_loss(obj_pred_confs, obj_iou_target, reduction='sum')

            noobj_pred_confs = pred_confs[iou_mask == 1]
            noobj_iou_target = iou_target[iou_mask == 1]
            noobj_iou_loss = F.mse_loss(noobj_pred_confs, noobj_iou_target, reduction='sum')

            # --------------------------------------
            # class loss
            # ignore the gradient of noobject's target
            pred_probs = pred_probs[class_mask > 0]
            class_target = class_target[class_mask > 0]
            class_loss = F.cross_entropy(pred_probs, class_target, reduction='sum')

            # # calculate the loss, normalized by batch size.
            # loss = box_loss * self.coord_scale + \
            #        obj_iou_loss * self.obj_scale + noobj_iou_loss * self.noobj_scale + \
            #        class_loss * self.class_scale
            # return loss / B

            # calculate the loss, normalized by batch size.
            lbox += box_loss * self.coord_scale
            lobj += obj_iou_loss * self.obj_scale + noobj_iou_loss * self.noobj_scale
            lcls += class_loss * self.class_scale

        # exit(1)
        return (lbox + lobj + lcls), torch.cat((lbox, lobj, lcls)).detach()
