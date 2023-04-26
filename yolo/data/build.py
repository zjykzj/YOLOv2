# -*- coding: utf-8 -*-

"""
@date: 2022/12/14 上午11:19
@file: build.py
@author: zj
@description: 
"""

import os
from typing import Dict
from argparse import Namespace

import torch

from .dataset.vocdataset import VOCDataset
from .transform import Transform
from .evaluate.vocevaluator import VOCEvaluator


def build_data(args: Namespace, cfg: Dict):
    # 创建转换器
    train_transform = Transform(cfg, is_train=True)
    val_transform = Transform(cfg, is_train=False)

    # 创建数据集
    data_type = cfg['DATA']['TYPE']
    max_det_num = cfg['DATA']['MAX_NUM_LABELS']
    if 'PASCAL VOC' == data_type:
        train_dataset_name = cfg['TRAIN']['DATASET_NAME']
        train_img_size = cfg['TRAIN']['IMGSIZE']
        train_dataset = VOCDataset(root=args.data,
                                   name=train_dataset_name,
                                   transform=train_transform,
                                   target_transform=None,
                                   target_size=train_img_size,
                                   max_det_nums=max_det_num
                                   )
        test_dataset_name = cfg['TEST']['DATASET_NAME']
        test_img_size = cfg['TEST']['IMGSIZE']
        val_dataset = VOCDataset(root=args.data,
                                 name=test_dataset_name,
                                 transform=val_transform,
                                 target_transform=None,
                                 target_size=test_img_size,
                                 max_det_nums=max_det_num
                                 )

        VOCdevkit_dir = os.path.join(args.data, cfg['TEST']['VOC'])
        year = cfg['TEST']['YEAR']
        split = cfg['TEST']['SPLIT']
        val_evaluator = VOCEvaluator(val_dataset.classes, VOCdevkit_dir, year=year, split=split)
    else:
        raise ValueError(f"{data_type} doesn't supports")

    # 创建采样器
    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    # 创建加载器
    collate_fn = torch.utils.data.default_collate
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['DATA']['BATCH_SIZE'], shuffle=(train_sampler is None),
        num_workers=cfg['DATA']['WORKERS'], pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True, sampler=None)

    return train_sampler, train_loader, val_loader, val_evaluator
