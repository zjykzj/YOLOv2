# -*- coding: utf-8 -*-

"""
@date: 2023/3/28 下午3:20
@file: vocdataset.py
@author: zj
@description: 
"""
import os
import cv2
import glob
import copy
import random

import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from . import KEY_IMAGE_ID, KEY_TARGET, KEY_IMAGE_INFO
from ..transform import Transform
from yolo.util.box_utils import label2yolobox


class VOCDataset(Dataset):
    classes = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self,
                 root: str,
                 name: str,
                 train: bool = True,
                 transform: Transform = None,
                 target_transform: Transform = None,
                 target_size: int = 416,
                 max_det_nums: int = 50):
        self.root = root
        self.name = name
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.target_size = target_size
        self.max_det_nums = max_det_nums

        image_dir = os.path.join(root, name, 'images')
        label_dir = os.path.join(root, name, 'labels')

        self.image_path_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        self.label_path_list = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
        assert len(self.image_path_list) == len(self.label_path_list)

        label_list = list()
        for image_path, label_path in zip(self.image_path_list, self.label_path_list):
            img_name = os.path.basename(image_path).rstrip('.jpg')
            label_name = os.path.basename(label_path).rstrip('.txt')
            assert img_name == label_name

            image = cv2.imread(image_path)
            img_h, img_w = image.shape[:2]

            sub_label_list = list()
            # [[cls_id, x_center, y_center, box_w, box_h], ]
            # The coordinate size is relative to the width and height of the image
            labels = np.loadtxt(label_path, delimiter=' ', dtype=float)
            if len(labels.shape) == 1:
                labels = [labels]
            for cls_id, xc, yc, box_w, box_h in labels:
                x_min = (xc - 0.5 * box_w) * img_w
                y_min = (yc - 0.5 * box_h) * img_h
                assert x_min >= 0 and y_min >= 0

                box_w = box_w * img_w
                box_h = box_h * img_h
                assert box_w < img_w and box_h < img_h

                # 转换成原始大小，方便后续图像预处理阶段进行转换和调试
                sub_label_list.append([cls_id, x_min, y_min, box_w, box_h])
            label_list.append(np.array(sub_label_list, dtype=float))

        self.label_list = label_list
        self.num_classes = len(self.classes)

    def _getdata(self, index=None):
        if index is None:
            index = random.choice(range(len(self.image_path_list)))

        image_path = self.image_path_list[index]
        image = cv2.imread(image_path)

        labels = copy.deepcopy(self.label_list[index])

        return index, image_path, image, labels

    def __getitem__(self, index) -> T_co:
        index, image_path, image, labels = self._getdata(index)

        # src_img = copy.deepcopy(image)
        # for (cls_id, x_min, y_min, box_w, box_h) in labels:
        #     cv2.rectangle(src_img, (int(x_min), int(y_min)), (int(x_min + box_w), int(y_min + box_h)),
        #                   (255, 255, 255), 1)
        # cv2.imshow('src_img', src_img)

        img_info = None
        if self.transform is not None:
            image_list = list()
            label_list = list()
            image_list.append(image)
            label_list.append(labels)

            if self.train and self.transform.is_mosaic:
                for _ in range(3):
                    while True:
                        _, _, sub_image, sub_labels = self._getdata()
                        if len(sub_labels) > 0:
                            break

                    image_list.append(sub_image)
                    label_list.append(sub_labels)

            image, labels, img_info = self.transform(image_list, label_list, self.target_size)

        # dst_img = copy.deepcopy(image).astype(np.uint8)
        # dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)
        # for (cls_id, x_min, y_min, box_w, box_h) in labels:
        #     cv2.rectangle(dst_img, (int(x_min), int(y_min)), (int(x_min + box_w), int(y_min + box_h)),
        #                   (255, 255, 255), 1)
        # cv2.imshow('dst_img', dst_img)
        # cv2.waitKey(0)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous() / 255

        target = self.build_target(labels)

        if self.train:
            return image, target
        else:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            target = {
                KEY_TARGET: target,
                KEY_IMAGE_INFO: img_info,
                KEY_IMAGE_ID: image_name
            }
            return image, target

    def build_target(self, labels):
        """
        :param bboxes: [[cls_id, x1, y1, box_w, box_h], ...]
        :return:
        """
        if len(labels) > 0:
            # 将数值缩放到[0, 1]区间
            labels[..., 1:] = labels[..., 1:] / self.target_size
            # [x1, y1, w, h] -> [xc, yc, w, h]
            labels[..., 1:] = label2yolobox(labels[..., 1:])

        target = torch.zeros((self.max_det_nums, 5))
        for i, label in enumerate(labels[:self.max_det_nums]):
            target[i, :] = torch.from_numpy(label)

        return target

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return super().__add__(other)

    def __len__(self):
        return len(self.image_path_list)

    def set_img_size(self, img_size):
        self.target_size = img_size

    def get_img_size(self):
        return self.target_size
