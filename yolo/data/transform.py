# -*- coding: utf-8 -*-

"""
@date: 2023/1/9 下午2:35
@file: transform.py
@author: zj
@description: 
"""
from typing import Dict, List

import cv2
import torch
import copy
import random

import numpy as np
from numpy import ndarray

from yolo.util.box_utils import xyxy2xywh, xywh2xyxy


def bgr2rgb(img: ndarray, is_rgb=True):
    if is_rgb:
        # return img[:, :, ::-1]
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        return img


def rect_intersection(a, b):
    minx = max(a[0], b[0])
    miny = max(a[1], b[1])

    maxx = min(a[2], b[2])
    maxy = min(a[3], b[3])
    return [minx, miny, maxx, maxy]


def crop_and_pad(src_img: ndarray, labels: ndarray, jitter_ratio: float = 0.3, ):
    # Base info
    src_h, src_w = src_img.shape[:2]

    jitter_h, jitter_w = np.array(np.array([src_h, src_w]) * jitter_ratio, dtype=int)
    crop_left = random.randint(-jitter_w, jitter_w)
    crop_right = random.randint(-jitter_w, jitter_w)
    crop_top = random.randint(-jitter_h, jitter_h)
    crop_bottom = random.randint(-jitter_h, jitter_h)

    crop_h = src_h - crop_top - crop_bottom
    crop_w = src_w - crop_left - crop_right
    assert crop_h > 1 and crop_w > 1

    # x1,y1,x2,y2
    crop_rect = [crop_left, crop_top, crop_left + crop_w, crop_top + crop_h]
    img_rect = [0, 0, src_w, src_h]

    intersection_rect = rect_intersection(crop_rect, img_rect)
    intersection_rect_w = intersection_rect[2] - intersection_rect[0]
    intersection_rect_h = intersection_rect[3] - intersection_rect[1]
    # x1,y1,x2,y2
    dst_intersection_rect = [max(0, -crop_left),
                             max(0, -crop_top),
                             max(0, -crop_left) + intersection_rect_w,
                             max(0, -crop_top) + intersection_rect_h]
    assert (dst_intersection_rect[3] - dst_intersection_rect[1]) == (intersection_rect[3] - intersection_rect[1])
    assert (dst_intersection_rect[2] - dst_intersection_rect[0]) == (intersection_rect[2] - intersection_rect[0])

    # Image Crop and Pad
    crop_img = np.zeros([crop_h, crop_w, 3])
    crop_img[:, :, ] = np.mean(src_img, axis=(0, 1))
    # crop_img[dst_y1:dst_y2, dst_x1:dst_x2] = src_img[y1:y2, x1:x2]
    crop_img[dst_intersection_rect[1]:dst_intersection_rect[3], dst_intersection_rect[0]:dst_intersection_rect[2]] = \
        src_img[intersection_rect[1]:intersection_rect[3], intersection_rect[0]:intersection_rect[2]]

    # BBoxes Crop and Pad
    # 如果真值边界框数目为0，那么返回
    if len(labels) != 0:
        # [cls_id, x1, y1, x2, y2]
        assert len(labels[0]) == 5
        # 随机打乱真值边界框
        np.random.shuffle(labels)
        # 原始图像的边界框坐标基于抖动调整坐标系
        # x_min / x_max
        labels[:, 1] -= crop_left
        labels[:, 3] -= crop_left
        # y_min / y_max
        labels[:, 2] -= crop_top
        labels[:, 4] -= crop_top

        # 精度截断
        # 设置x0, x1的最大最小值
        labels[:, 1] = np.clip(labels[:, 1], 0, crop_w - 1)
        labels[:, 3] = np.clip(labels[:, 3], 0, crop_w - 1)
        # 设置y0，y1的最大最小值
        labels[:, 2] = np.clip(labels[:, 2], 0, crop_h - 1)
        labels[:, 4] = np.clip(labels[:, 4], 0, crop_h - 1)

        # 找出x0==x1，或者y0==y1的边界框
        # 也就是说，边界框经过抖动和截断后变成了一条线
        out_box = list(
            np.where(
                (labels[:, 1] == labels[:, 3]) |
                (labels[:, 2] == labels[:, 4])
            )[0]
        )
        list_box = list(range(labels.shape[0]))
        # 移除这种边界框
        for i in out_box:
            list_box.remove(i)
        # 获取剩余边界框
        labels = labels[list_box]

    crop_info = [crop_left, crop_right, crop_top, crop_bottom, crop_w, crop_h]
    return crop_img, labels, crop_info


def left_right_flip(img, labels, crop_info, is_flip=True):
    assert len(img.shape) == 3 and img.shape[2] == 3

    is_flip = is_flip and np.random.randn() > 0.5
    if is_flip:
        # [H, W, C]
        img = np.flip(img, axis=1).copy()
        h, w = img.shape[:2]

        if len(labels) > 0:
            # 左右翻转，所以y值不变，变换x值
            tmp_x = w - labels[:, 1]
            labels[:, 1] = w - labels[:, 3]
            labels[:, 3] = tmp_x

        crop_w, crop_h = crop_info[4:6]
        crop_info[4] = crop_h
        crop_info[5] = crop_w

    return img, labels, crop_info, is_flip


def image_resize(img, labels, dst_size):
    sized_img = cv2.resize(img, (dst_size, dst_size), cv2.INTER_LINEAR)

    img_h, img_w = img.shape[:2]

    if len(labels) > 0:
        # 转换抖动图像上的边界框坐标到网络输入图像的边界框坐标
        # dst_bbox / bbox = dst_size / img_size
        # dst_bbox = bbox * (dst_size / img_size)
        #
        # x_min / x_max
        labels[..., 1] *= (dst_size / img_w)
        labels[..., 3] *= (dst_size / img_w)
        # y_min / y_max
        labels[..., 2] *= (dst_size / img_h)
        labels[..., 4] *= (dst_size / img_h)

    return sized_img, labels


def rand_uniform_strong(min, max):
    """
    随机均匀增强
    """
    if min > max:
        swap = min
        min = max
        max = swap
    return random.random() * (max - min) + min


def rand_scale(s):
    """
    随机缩放，放大或者缩小
    """
    scale = rand_uniform_strong(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale


def color_dithering(img, hue, saturation, exposure, is_jitter=True):
    """
    img: 图像 [H, W, 3]
    hue: 色调
    saturation: 饱和度
    exposure: 曝光度
    """
    if is_jitter:
        # 色度、饱和度、曝光度
        dhue = rand_uniform_strong(-hue, hue)
        dsat = rand_scale(saturation)
        dexp = rand_scale(exposure)

        src_dtype = img.dtype
        img = img.astype(np.float32)

        # HSV augmentation
        # 先转换到HSV颜色空间，然后手动调整饱和度、亮度和色度，最后转换成为RGB颜色空间
        # cv2.COLOR_BGR2HSV, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2BGR, cv2.COLOR_HSV2RGB
        if dsat != 1 or dexp != 1 or dhue != 0:
            if img.shape[2] >= 3:
                hsv_src = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # RGB to HSV
                # Solve: https://github.com/Tianxiaomo/pytorch-YOLOv4/issues/427
                hsv = list(cv2.split(hsv_src))
                hsv[1] *= dsat
                hsv[2] *= dexp
                hsv[0] += 179 * dhue
                hsv_src = cv2.merge(hsv)
                # HSV to RGB (the same as previous)
                img = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)
            else:
                img *= dexp

        img.astype(src_dtype)
    return img


def filter_truth(labels, dx, dy, sx, sy, xd, yd):
    if len(labels) <= 0:
        return labels

    assert dx >= 0 and dy >= 0
    # 图像抖动后的边界框坐标
    # x_min / x_max
    labels[:, 1] -= dx
    labels[:, 3] -= dx
    # y_min / y_max
    labels[:, 2] -= dy
    labels[:, 4] -= dy

    assert sx > 0 and sy > 0
    # 边界框大小不能超出裁剪区域
    # x_min / x_max
    labels[:, 1] = np.clip(labels[:, 1], 0, sx)
    labels[:, 3] = np.clip(labels[:, 3], 0, sx)
    # y_min / y_max
    labels[:, 2] = np.clip(labels[:, 2], 0, sy)
    labels[:, 4] = np.clip(labels[:, 4], 0, sy)

    # 过滤截断后不存在的边界框
    out_box = list(np.where(((labels[:, 2] == sy) & (labels[:, 4] == sy)) |
                            ((labels[:, 1] == sx) & (labels[:, 2] == sx)) |
                            ((labels[:, 2] == 0) & (labels[:, 4] == 0)) |
                            ((labels[:, 1] == 0) & (labels[:, 3] == 0)))[0])
    list_box = list(range(labels.shape[0]))
    for i in out_box:
        list_box.remove(i)
    labels = labels[list_box]

    assert xd >= 0 and yd >= 0
    # mosaic后的图像左上角坐标
    # x_min / x_max
    labels[:, 1] += xd
    labels[:, 3] += xd
    # y_min / y_max
    labels[:, 2] += yd
    labels[:, 4] += yd

    return labels


def blend_mosaic(out_img, img, labels, cut_x, cut_y, mosaic_idx, crop_info):
    crop_left, crop_right, crop_top, crop_bottom, crop_w, crop_h = crop_info[:6]
    img_h, img_w = img.shape[:2]

    # left_shift / top_shift / right_shift / bottom_shift > 0
    # dst_left / crop_left = dst_w / crop_w
    # dst_left = crop_left * (dst_w / crop_w)
    left_shift = int(min(cut_x, max(0, (-int(crop_left) * img_w / crop_w))))
    top_shift = int(min(cut_y, max(0, (-int(crop_top) * img_h / crop_h))))
    right_shift = int(min((img_w - cut_x), max(0, (-int(crop_right) * img_w / crop_w))))
    bottom_shift = int(min((img_h - cut_y), max(0, (-int(crop_bottom) * img_h / crop_h))))

    left_shift = min(left_shift, img_w - cut_x)
    top_shift = min(top_shift, img_h - cut_y)
    right_shift = min(right_shift, cut_x)
    bottom_shift = min(bottom_shift, cut_y)

    if mosaic_idx == 0:
        # 左上角贴图，大小：[h, w]=[cut_y, cut_x]，左上角坐标：[x, y]=[0, 0]
        # 原图裁剪左上角坐标：[x, y]=[left_shift, top_shift]
        labels = filter_truth(labels, left_shift, top_shift, cut_x, cut_y, 0, 0)
        out_img[:cut_y, :cut_x] = img[top_shift:top_shift + cut_y, left_shift:left_shift + cut_x]
    if mosaic_idx == 1:
        # 右上角贴图，大小：[h, w]=[cut_y, w-cut_x]，左上角坐标：[x,y]=[cut_x, 0]
        # 原图裁剪左上角坐标：[x, y]=[cut_x-right_shift, top_shift]
        labels = filter_truth(labels, cut_x - right_shift, top_shift, img_w - cut_x, cut_y, cut_x, 0)
        out_img[:cut_y, cut_x:] = img[top_shift:top_shift + cut_y, cut_x - right_shift:img_w - right_shift]
    if mosaic_idx == 2:
        # 左下角
        labels = filter_truth(labels, left_shift, cut_y - bottom_shift, cut_x, img_h - cut_y, 0, cut_y)
        out_img[cut_y:, :cut_x] = img[cut_y - bottom_shift:img_h - bottom_shift, left_shift:left_shift + cut_x]
    if mosaic_idx == 3:
        # 右下角
        labels = filter_truth(labels, cut_x - right_shift, cut_y - bottom_shift,
                              img_w - cut_x, img_h - cut_y, cut_x, cut_y)
        out_img[cut_y:, cut_x:] = img[cut_y - bottom_shift:img_h - bottom_shift,
                                  cut_x - right_shift:img_w - right_shift]

    return out_img, labels


class Transform(object):
    """
    图像+标签转换

    训练阶段：颜色通道转换（BGR->RGB）、随机裁剪+填充、随机翻转、图像缩放、颜色抖动、mosaic
    验证阶段：颜色通道转换（BGR->RGB）、图像缩放
    """

    def __init__(self, cfg: Dict, is_train: bool = True):
        self.is_train = is_train

        # RGB
        self.is_rgb = cfg['AUGMENTATION']['RGB']
        # crop
        self.jitter_ratio = cfg['AUGMENTATION']['JITTER']
        # self.jitter_ratio = 0.
        # flip
        self.is_flip = cfg['AUGMENTATION']['RANDOM_HORIZONTAL_FLIP']
        # self.is_flip = False
        # color jitter
        self.color_jitter = cfg['AUGMENTATION']['COLOR_DITHERING']
        # self.color_jitter = False
        self.hue = cfg['AUGMENTATION']['HUE']
        self.saturation = cfg['AUGMENTATION']['SATURATION']
        self.exposure = cfg['AUGMENTATION']['EXPOSURE']
        # mosaic
        self.is_mosaic = cfg['AUGMENTATION']['IS_MOSAIC']
        # self.is_mosaic = False
        self.min_offset = cfg['AUGMENTATION']['MIN_OFFSET']

    def _get_train_item(self, img_list: List[ndarray], label_list: List[ndarray], img_size: int):
        # 指定结果图像的宽/高
        out_img = np.zeros([img_size, img_size, 3])
        # 在输出图像上的真值边界框可以有多个
        out_labels = []

        # 进行随机裁剪，随机生成裁剪图像的起始坐标
        # 坐标x取值在[0.2*w, 0.8*w]之间，坐标y同理
        cut_x = random.randint(int(img_size * self.min_offset), int(img_size * (1 - self.min_offset)))
        cut_y = random.randint(int(img_size * self.min_offset), int(img_size * (1 - self.min_offset)))

        for idx, (img, labels) in enumerate(zip(img_list, label_list)):
            assert len(labels) == 0 or labels.shape[1] == 5
            assert len(img.shape) == 3 and img.shape[2] == 3
            if len(labels) > 0:
                labels[..., 1:] = xywh2xyxy(labels[..., 1:], is_center=False)

            # BGR -> RGB
            img = bgr2rgb(img, is_rgb=self.is_rgb)
            # 随机裁剪 + 填充
            img, labels, crop_info = crop_and_pad(img, labels, self.jitter_ratio)
            # 随机翻转
            img, labels, crop_info, is_flip = left_right_flip(img, labels, crop_info, is_flip=self.is_flip)
            # 图像缩放
            img, labels = image_resize(img, labels, img_size)
            # 最后进行颜色抖动
            img = color_dithering(img, self.hue, self.saturation, self.exposure, is_jitter=self.color_jitter)

            if self.is_mosaic:
                assert len(img_list) == 4 and len(label_list) == 4
                out_img, labels = blend_mosaic(out_img, img, labels, cut_x, cut_y, idx, crop_info)
                if len(labels) > 0:
                    out_labels.append(labels)
            else:
                assert len(img_list) == 1 and len(label_list) == 1
                out_img = img
                out_labels = labels

        if self.is_mosaic and len(out_labels) > 0:
            out_labels = np.concatenate(out_labels, axis=0)
        if len(out_labels) > 0:
            out_labels[..., 1:] = np.clip(out_labels[..., 1:], 0, img_size - 1)
            out_labels[..., 1:] = xyxy2xywh(out_labels[..., 1:], is_center=False)

        img_info = list()
        return out_img, out_labels, img_info

    def _get_val_item(self, img_list: List[ndarray], label_list: List[ndarray], img_size: int):
        assert len(img_list) == 1 and len(label_list) == 1
        src_img = img_list[0]
        src_labels = label_list[0]

        # labels: [cls_id, x_min, y_min, box_w, box_h]
        assert len(src_labels) == 0 or len(src_labels[0]) == 5
        if len(src_labels) > 0:
            src_labels[..., 1:] = xywh2xyxy(src_labels[..., 1:], is_center=False)

        dst_img = bgr2rgb(src_img)
        # 图像缩放
        dst_img, dst_labels = image_resize(dst_img, src_labels, img_size)

        src_img_h, src_img_w = src_img.shape[:2]
        dst_img_h, dst_img_w = dst_img.shape[:2]
        dx = 0
        dy = 0
        img_info = [src_img_h, src_img_w, dst_img_h, dst_img_w, dx, dy]

        if len(dst_labels) > 0:
            dst_labels[..., 1:] = np.clip(dst_labels[..., 1:], 0, img_size - 1)
            dst_labels[..., 1:] = xyxy2xywh(dst_labels[..., 1:], is_center=False)
        return dst_img, dst_labels, img_info

    def __call__(self, img_list: List[ndarray], label_list: List[ndarray], img_size: int):
        """
        bboxes_list: [bboxes, ...]
        bboxes: [[cls_id, x_min, y_min, box_w, box_h], ...]
        """
        if self.is_train:
            out_img, out_labels, img_info = self._get_train_item(img_list, label_list, img_size)
        else:
            out_img, out_labels, img_info = self._get_val_item(img_list, label_list, img_size)

        return out_img, out_labels, img_info
