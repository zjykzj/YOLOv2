# -*- coding: utf-8 -*-

"""
@date: 2023/3/28 下午3:20
@file: vocdataset.py
@author: zj
@description: 
"""
import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from ..transform import Transform


def coco2yolobox(labels):
    # x1/y1/w/h -> x1/y1/x2/y2
    x1 = labels[:, 0]
    y1 = labels[:, 1]
    x2 = (labels[:, 0] + labels[:, 2])
    y2 = (labels[:, 1] + labels[:, 3])

    # x1/y1/x2/y2 -> xc/yc/w/h
    labels[:, 0] = ((x1 + x2) / 2)
    labels[:, 1] = ((y1 + y2) / 2)
    return labels


def xywh2xyxy(boxes):
    boxes[:, 2] = (boxes[:, 0] + boxes[:, 2])
    boxes[:, 3] = (boxes[:, 1] + boxes[:, 3])
    return boxes


class VOCDataset(Dataset):

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

        box_list = list()
        label_list = list()
        for image_path, label_path in zip(self.image_path_list, self.label_path_list):
            img_name = os.path.basename(image_path).rstrip('.jpg')
            label_name = os.path.basename(label_path).rstrip('.txt')
            assert img_name == label_name

            image = cv2.imread(image_path)
            img_h, img_w = image.shape[:2]

            sub_box_list = list()
            sub_label_list = list()
            # [[cls_id, x_center, y_center, box_w, box_h], ]
            # The coordinate size is relative to the width and height of the image
            boxes = np.loadtxt(label_path, delimiter=' ', dtype=float)
            if len(boxes.shape) == 1:
                boxes = [boxes]
            for label, xc, yc, box_w, box_h in boxes:
                x_min = (xc - 0.5 * box_w) * img_w
                y_min = (yc - 0.5 * box_h) * img_h
                box_w = box_w * img_w
                box_h = box_h * img_h
                sub_box_list.append([x_min, y_min, box_w, box_h])
                sub_label_list.append(int(label))
            box_list.append(np.array(sub_box_list))
            label_list.append(sub_label_list)

        self.box_list = np.array(box_list, dtype=object)
        self.label_list = label_list
        self.num_classes = 20

    def __getitem__(self, index) -> T_co:
        image_path = self.image_path_list[index]
        boxes = self.box_list[index]
        labels = self.label_list[index]

        image = cv2.imread(image_path)

        # src_img = copy.deepcopy(image)
        # for box in boxes:
        #     x_min, y_min, box_w, box_h = box
        #     cv2.rectangle(src_img, (int(x_min), int(y_min)), (int(x_min + box_w), int(y_min + box_h)),
        #                   (255, 255, 255), 1)
        # cv2.imshow('src_img', src_img)

        image, boxes, img_info = self.transform(image, boxes, self.target_size)

        # dst_img = copy.deepcopy(image).astype(np.uint8)
        # dst_img = cv2.cvtColor(dst_img, cv2.COLOR_RGB2BGR)
        # for box in boxes:
        #     x_min, y_min, box_w, box_h = box
        #     cv2.rectangle(dst_img, (int(x_min), int(y_min)), (int(x_min + box_w), int(y_min + box_h)),
        #                   (255, 255, 255), 1)
        # cv2.imshow('dst_img', dst_img)
        # cv2.waitKey(0)

        # 数据预处理
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous() / 255

        boxes = boxes / self.target_size
        # boxes = coco2yolobox(boxes)
        boxes = xywh2xyxy(boxes)
        target = self.build_target(boxes, labels)
        if self.train:
            return image, target
        else:
            return image, target, np.array(img_info)

    def build_target(self, boxes, labels):
        """
        :param boxes: [[xc, yc, box_w, box_h], ...]
        :param labels: [box1_cls_idx, ...]
        :return:
        """
        target = torch.zeros((self.max_det_nums, 5))

        for i, (box, label) in enumerate(zip(boxes[:self.max_det_nums], labels[:self.max_det_nums])):
            target[i, :4] = torch.from_numpy(box)
            target[i, 4] = label

        return target

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return super().__add__(other)

    def __len__(self):
        return len(self.image_path_list)


if __name__ == '__main__':
    random.seed(10)
    root = '/home/zj/data/voc'
    name = 'voc2yolov5-val'

    # test_dataset = VOCDataset(root, name, S=7, B=2, train=False, transform=Transform(is_train=False))
    # image, target = test_dataset.__getitem__(300)
    # print(image.shape, target.shape)

    train_dataset = VOCDataset(root, name, train=True, transform=Transform(is_train=True))

    for i in [31, 62, 100, 633]:
        image, target = train_dataset.__getitem__(i)
        print(i, image.shape, target.shape)