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


def generate_all_grid_anchors(anchors: Tensor, F_H: int = 13, F_W: int = 13):
    assert tuple(anchors.shape) == (5, 2)

    # shift_y: [H, W] [[0, 0, 0, ...], [...]]
    # shift_x: [H, W] [[0, 1, 2, ...], [...]]
    shift_y, shift_x = torch.meshgrid([torch.arange(0, F_H), torch.arange(0, F_W)])

    # [H*W, 2]
    centers = torch.cat([shift_x.reshape(-1), shift_y.reshape(-1)], dim=-1)

    K = F_H * F_W
    A = len(anchors)
    # [H*W, A, 4] -> [H*W*A, 4]
    all_grid_anchors = torch.cat([centers.view(K, 1, 2).expand(K, A, 2),
                                  anchors.view(1, A, 2).expand(K, A, 2)], dim=-1).reshape(-1, 4)

    return all_grid_anchors


def box_transform()
    pass


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
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


class YOLOv2Loss(nn.Module):

    def __init__(self, anchors, num_anchors=5, num_classes=20, ignore_thresh=0.75):
        super(YOLOv2Loss, self).__init__()
        self.anchors = torch.FloatTensor(anchors)
        self.num_anchors = num_anchors
        assert tuple(self.anchors.shape) == (num_anchors, 2)
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh

    def build_target(self, outputs: Tensor, targets: Tensor):
        B, _, F_H, F_W = outputs.shape[:4]

        # [B, C, H, W] -> [B, H, W, C]
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        # [B, H, W, C] -> [B, H, W, num_anchors*4] -> [B, H, W, num_anchors, 4]
        pred_box_offsets = outputs[..., :self.num_anchors * 4].reshape(B, H, W, self.num_anchors, 4)
        # [B, H, W, C] -> [B, H, W, num_anchors]
        pred_confs = outputs[..., self.num_anchors * 4:self.num_anchors * 5]
        # [B, H, W, C] -> [B, H, W, num_classes]
        pred_cls_probs = outputs[..., self.num_anchors * 5:]

        # 坐标转换
        # b_x = sigmoid(t_x) + c_x
        # b_y = sigmoid(t_y) + c_y
        # b_w = p_w * e^t_w
        # b_h = p_h * e^t_h
        pred_box_offsets[..., :2] = torch.sigmoid(pred_box_offsets[..., :2])
        pred_box_offsets[..., 2:] = torch.exp(pred_box_offsets[..., 2:])
        # 分类概率压缩
        pred_cls_probs = torch.softmax(pred_cls_probs, dim=-1)

        # [B, H*W, num_anchors, 1]
        iou_target = outputs.new_zeros((B, F_H * F_W, self.num_anchors, 1))
        iou_mask = outputs.new_ones((B, F_H * F_W, self.num_anchors, 1))
        #
        box_target = outputs.new_zeros((B, F_H * F_W, self.num_anchors, 4))
        box_mask = outputs.new_zeros((B, F_H * F_W, self.num_anchors, 1))
        #
        class_target = outputs.new_zeros((B, F_H * F_W, self.num_anchors, 1))
        class_mask = outputs.new_zeros((B, F_H * F_W, self.num_anchors, 1))

        # 创建每个网格的锚点框数据
        all_grid_anchors = generate_all_grid_anchors(self.anchors, F_H, F_W)
        all_grid_anchors_xcycwh = all_grid_anchors.clone()
        all_grid_anchors_xcycwh[..., :2] += 0.5

        # [B, num_max_det, 5] -> [B, num_max_det, 4] -> [B, num_max_det] -> [B]
        gt_num_objs = (targets[..., :4].sum(dim=2) > 0).sum(dim=1)

        for bi in range(B):
            # 获取该图像有效的标注框数目
            num_obj = gt_num_objs[bi]
            # [B, num_max_det, 5] -> [num_max_det, 4]
            gt_boxes = targets[bi, :num_obj, :4]
            # [B, num_max_det, 5] -> [num_max_det]
            gt_labels = targets[bi, :num_obj, 4]

            # 计算该图像标注框与锚点框IoU



            # 每个标注框匹配一个锚点框
            for t in range(num_obj):
                gt_box = gt_boxes[t]
                gt_label = gt_labels[t]
                # 计算该标注框所处网格下标
                cell_x, cell_y = torch.floor(gt_box[:2])
                # 网格列表下标
                cell_idx = cell_y * F_W + cell_x

    def forward(self, outputs, targets):
        # [B, num_anchors * (5 + num_classes), F_H, F_W]
        B, C, H, W = outputs.shape[:4]
        assert C == (self.num_anchors * (5 + self.num_classes))
        F_Size = outputs.shape[2]

        # [B, C, H, W] -> [B, H, W, C]
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        # [B, H, W, C] -> [B, H, W, num_anchors*4] -> [B, H, W, num_anchors, 4]
        pred_box_offsets = outputs[..., :self.num_anchors * 4].reshape(B, H, W, self.num_anchors, 4)
        # [B, H, W, C] -> [B, H, W, num_anchors]
        pred_confs = outputs[..., self.num_anchors * 4:self.num_anchors * 5]
        # [B, H, W, C] -> [B, H, W, num_classes]
        pred_cls_probs = outputs[..., self.num_anchors * 5:]

        # 坐标转换
        # b_x = sigmoid(t_x) + c_x
        # b_y = sigmoid(t_y) + c_y
        # b_w = p_w * e^t_w
        # b_h = p_h * e^t_h
        pred_box_offsets[..., :2] = torch.sigmoid(pred_box_offsets[..., :2])
        pred_box_offsets[..., 2:] = torch.exp(pred_box_offsets[..., 2:])
        # 分类概率压缩
        pred_cls_probs = torch.softmax(pred_cls_probs, dim=-1)

        # anchors
        shift_y, shift_x = torch.meshgrid([torch.range(0, H), torch.range(0, W)])

        for bi in range(B):
