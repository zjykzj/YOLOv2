# -*- coding: utf-8 -*-

"""
@date: 2023/4/26 下午9:42
@file: dataset.py
@author: zj
@description: 
"""

import random

from yolo.data.dataset.vocdataset import VOCDataset
from yolo.data.transform import Transform


def test_train(root, cfg):
    print("=> Train")
    name = 'voc2yolov5-train'
    train_dataset = VOCDataset(root, name, train=True, transform=Transform(cfg, is_train=True))
    print("Total len:", len(train_dataset))
    for i in [31, 62, 100, 633]:
        image, target = train_dataset.__getitem__(i)
        print(i, image.shape, target.shape)


def test_val(root, cfg):
    print("=> Val")
    name = 'voc2yolov5-val'
    val_dataset = VOCDataset(root, name, train=False, transform=Transform(cfg, is_train=False))
    print("Total len:", len(val_dataset))

    # i = 170
    # image, target = val_dataset.__getitem__(i)
    # print(i, image.shape, target['target'].shape, len(target['img_info']), target['image_name'])

    # for i in [31, 62, 100, 166, 169, 170, 633]:
    #     image, target = val_dataset.__getitem__(i)
    #     print(i, image.shape, target['target'].shape, len(target['img_info']))
    for i in range(len(val_dataset)):
        image, target = val_dataset.__getitem__(i)
        print(i, image.shape, target['target'].shape, len(target['img_info']), target['image_name'])


if __name__ == '__main__':
    random.seed(10)
    root = '../datasets/voc'

    cfg_file = 'configs/yolov2_default.cfg'
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    # test_dataset = VOCDataset(root, name, S=7, B=2, train=False, transform=Transform(is_train=False))
    # image, target = test_dataset.__getitem__(300)
    # print(image.shape, target.shape)

    # test_train(root, cfg)
    test_val(root, cfg)
