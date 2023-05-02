# -*- coding: utf-8 -*-

"""
@date: 2023/4/26 下午9:42
@file: dataset.py
@author: zj
@description: 
"""

import random
import time

from yolo.data.dataset import KEY_TARGET, KEY_IMAGE_INFO, KEY_IMAGE_ID
from yolo.data.dataset.cocodataset import COCODataset
from yolo.data.dataset.vocdataset import VOCDataset
from yolo.data.transform import Transform


def test_voc_train():
    root = '../datasets/voc'

    cfg_file = 'configs/yolov2_voc.cfg'
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    # test_dataset = VOCDataset(root, name, S=7, B=2, train=False, transform=Transform(is_train=False))
    # image, target = test_dataset.__getitem__(300)
    # print(image.shape, target.shape)

    print("=> Pascal VOC Train")
    name = 'voc2yolov5-train'
    train_dataset = VOCDataset(root, name, train=True, transform=Transform(cfg, is_train=True))
    print("Total len:", len(train_dataset))
    # for i in [31, 62, 100, 633]:

    end = time.time()
    for i in range(len(train_dataset)):
        image, target = train_dataset.__getitem__(i)
        print(i, image.shape, target.shape)
    print(f"Avg one time: {(time.time() - end) / len(train_dataset)}")


def test_voc_val():
    root = '../datasets/voc'

    cfg_file = 'configs/yolov2_voc.cfg'
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    # test_dataset = VOCDataset(root, name, S=7, B=2, train=False, transform=Transform(is_train=False))
    # image, target = test_dataset.__getitem__(300)
    # print(image.shape, target.shape)

    print("=> Pascal VOC Val")
    name = 'voc2yolov5-val'
    val_dataset = VOCDataset(root, name, train=False, transform=Transform(cfg, is_train=False))
    print("Total len:", len(val_dataset))

    # i = 170
    # image, target = val_dataset.__getitem__(i)
    # print(i, image.shape, target['target'].shape, len(target['img_info']), target['image_name'])

    # for i in [31, 62, 100, 166, 169, 170, 633]:
    #     image, target = val_dataset.__getitem__(i)
    #     print(i, image.shape, target['target'].shape, len(target['img_info']))

    end = time.time()
    for i in range(len(val_dataset)):
        image, target = val_dataset.__getitem__(i)
        print(i, image.shape, target[KEY_TARGET].shape, len(target[KEY_IMAGE_INFO]), target[KEY_IMAGE_ID])
    print(f"Avg one time: {(time.time() - end) / len(val_dataset)}")


def test_coco_train():
    cfg_file = 'configs/yolov2_coco.cfg'
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    print("=> COCO Train")
    root = '../datasets/coco'
    transform = Transform(cfg, is_train=True)
    dataset = COCODataset(root, name='train2017', train=True, transform=transform, target_size=416)

    # img, target = dataset.__getitem__(333)
    # img, target = dataset.__getitem__(57756)
    # img, target = dataset.__getitem__(87564)
    # img, target = dataset.__getitem__(51264)

    end = time.time()
    for i in range(len(dataset)):
        image, target = dataset.__getitem__(i)
        print(i, image.shape, target.shape)
    print(f"Avg one time: {(time.time() - end) / len(dataset)}")


def test_coco_val():
    cfg_file = 'configs/yolov2_coco.cfg'
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    print("=> COCO Val")
    root = '../datasets/coco'
    transform = Transform(cfg, is_train=False)
    dataset = COCODataset(root, name='val2017', train=False, transform=transform, target_size=416)

    # img, target = dataset.__getitem__(333)
    # img, target = dataset.__getitem__(57756)
    # img, target = dataset.__getitem__(87564)
    # img, target = dataset.__getitem__(51264)

    end = time.time()
    for i in range(len(dataset)):
        image, target = dataset.__getitem__(i)
        print(i, image.shape, target.shape)
    print(f"Avg one time: {(time.time() - end) / len(dataset)}")


if __name__ == '__main__':
    random.seed(10)

    # test_voc_train()
    # test_voc_val()
    test_coco_train()
    test_coco_val()
