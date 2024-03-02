# -*- coding: utf-8 -*-

"""
@date: 2024/3/2 下午2:49
@file: create_anchors.py
@author: zj
@description: Create a list of anchor points with different numbers
"""

from utils.dataloaders import LoadImagesAndLabels
from utils.autoanchor import kmean_anchors


def create_dataset(path):
    imgsz = 640
    batch_size = 16
    stride = 32

    dataset = LoadImagesAndLabels(
        path,
        imgsz,
        batch_size,
        augment=False,  # augmentation
        hyp=None,  # hyperparameters
        rect=False,  # rectangular batches
        cache_images=False,
        single_cls=False,
        stride=int(stride),
        pad=0.0,
        image_weights=False,
        prefix='')

    return dataset, imgsz


if __name__ == '__main__':
    path = '../datasets/coco/train2017.txt'
    # path = '../datasets/coco/val2017.txt'
    dataset, imgsz = create_dataset(path)

    thr = 4.0
    # num anchors
    for na in range(1, 10):
        anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        print(f"na: {na} - anchors: \n{anchors}")
