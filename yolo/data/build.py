# -*- coding: utf-8 -*-

"""
@date: 2022/12/14 上午11:19
@file: build.py
@author: zj
@description: 
"""

from typing import Dict
from argparse import Namespace

import torch

from .vocdataset import VOCDataset
from .transform import Transform


def build_data(args: Namespace, cfg: Dict):
    # 创建转换器
    train_transform = Transform(cfg, is_train=True)
    val_transform = Transform(cfg, is_train=False)

    # 创建数据集
    train_dataset = VOCDataset(root=args.data,
                               name='train2017',
                               transform=train_transform,
                               target_transform=None,
                               target_size=416,
                               max_det_nums=50
                               )
    val_dataset = VOCDataset(root=args.data,
                             name='val2017',
                             transform=val_transform,
                             target_transform=None,
                             target_size=416,
                             max_det_nums=50
                             )

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

    return train_sampler, train_loader, val_loader
