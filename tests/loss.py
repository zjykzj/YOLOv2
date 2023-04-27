# -*- coding: utf-8 -*-

"""
@date: 2023/4/27 下午3:37
@file: anchors.py
@author: zj
@description: 
"""

import torch

from yolo.model.yololoss import YOLOv2Loss


def test_anchors():
    cfg_file = 'configs/yolov2_default.cfg'
    print(f"=> Test {cfg_file}")
    with open(cfg_file, 'r') as f:
        import yaml
        cfg = yaml.safe_load(f)

    anchors = torch.tensor(cfg['MODEL']['ANCHORS'])
    criterion = YOLOv2Loss(anchors,
                           num_classes=cfg['MODEL']['N_CLASSES'],
                           ignore_thresh=cfg['CRITERION']['IGNORE_THRESH'],
                           coord_scale=cfg['CRITERION']['COORD_SCALE'],
                           noobj_scale=cfg['CRITERION']['NOOBJ_SCALE'],
                           obj_scale=cfg['CRITERION']['OBJ_SCALE'],
                           class_scale=cfg['CRITERION']['CLASS_SCALE'],
                           )
    all_anchors = criterion.build_anchors(13)
    print(all_anchors.shape)


if __name__ == '__main__':
    test_anchors()
