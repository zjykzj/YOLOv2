# -*- coding: utf-8 -*-

"""
@date: 2022/12/14 上午11:19
@file: build.py
@author: zj
@description: 
"""

from typing import Dict
from argparse import Namespace

import os
import torch

from .dataset.vocdataset import VOCDataset
from .transform import Transform
from .evaluate.vocevaluator import VOCEvaluator


def build_data(cfg: Dict, data_root: str, is_train: bool = True, is_distributed: bool = False):
    data_type = cfg['DATA']['TYPE']
    max_det_num = cfg['DATA']['MAX_NUM_LABELS']

    sampler = None
    transform = Transform(cfg, is_train=is_train)
    collate_fn = torch.utils.data.default_collate

    if is_train:
        if 'PASCAL VOC' == data_type:
            train_dataset_name = cfg['TRAIN']['DATASET_NAME']
            train_img_size = cfg['TRAIN']['IMGSIZE']
            dataset = VOCDataset(root=data_root,
                                 name=train_dataset_name,
                                 train=is_train,
                                 transform=transform,
                                 target_transform=None,
                                 target_size=train_img_size,
                                 max_det_nums=max_det_num
                                 )
        else:
            raise ValueError(f"{data_type} doesn't supports")

        if is_distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=cfg['DATA']['BATCH_SIZE'],
                                                 shuffle=(sampler is None),
                                                 num_workers=cfg['DATA']['WORKERS'],
                                                 sampler=sampler,
                                                 collate_fn=collate_fn,
                                                 pin_memory=True
                                                 )

        return dataloader, sampler
    else:
        if 'PASCAL VOC' == data_type:
            test_dataset_name = cfg['TEST']['DATASET_NAME']
            test_img_size = cfg['TEST']['IMGSIZE']
            dataset = VOCDataset(root=data_root,
                                 name=test_dataset_name,
                                 train=is_train,
                                 transform=transform,
                                 target_transform=None,
                                 target_size=test_img_size,
                                 max_det_nums=max_det_num
                                 )

            VOCdevkit_dir = os.path.join(data_root, cfg['TEST']['VOC'])
            year = cfg['TEST']['YEAR']
            split = cfg['TEST']['SPLIT']
            val_evaluator = VOCEvaluator(dataset.classes, VOCdevkit_dir, year=year, split=split)
        else:
            raise ValueError(f"{data_type} doesn't supports")

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=cfg['DATA']['BATCH_SIZE'],
                                                 shuffle=False,
                                                 num_workers=cfg['DATA']['WORKERS'],
                                                 sampler=None,
                                                 collate_fn=collate_fn,
                                                 pin_memory=True,
                                                 )

        return dataloader, sampler, val_evaluator
