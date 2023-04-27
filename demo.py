# -*- coding: utf-8 -*-

"""
@date: 2023/4/23 上午10:01
@file: demo.py
@author: zj
@description: 
"""
from typing import List, Tuple, Dict

import cv2
import yaml
import os.path

import argparse
from argparse import Namespace

import numpy as np
from numpy import ndarray

import torch.cuda
from torch import Tensor
from torch.nn import Module

from yolo.model.yolov2 import YOLOv2
from yolo.data.dataset.vocdataset import VOCDataset
from yolo.data.transform import Transform
from yolo.util.utils import postprocess, yolobox2label


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv2 Demo.')
    parser.add_argument('cfg', type=str, default='configs/yolov2_default.cfg', help='Model configuration file.')
    parser.add_argument('ckpt', type=str, default=None, help='Path to the checkpoint file.')
    parser.add_argument('image', type=str, default=None, help='Path to image file')
    parser.add_argument('-c', '--conf-thresh', type=float, default=None, help='Confidence Threshold')
    parser.add_argument('-n', '--nms-thresh', type=float, default=None, help='NMS Threshold')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    return args, cfg


def image_preprocess(args: Namespace, cfg: Dict):
    transform = Transform(cfg, is_train=False)

    # BGR
    img = cv2.imread(args.image)
    img_raw = img.copy()

    imgsize = cfg['TEST']['IMGSIZE']
    img, _, img_info = transform(img, np.array([]), imgsize)
    # [H, W, C] -> [C, H, W]
    img = torch.from_numpy(img).permute(2, 0, 1).contiguous() / 255
    print("img:", img.shape)

    # 返回输入图像数据、原始图像数据、图像缩放前后信息
    return img, img_raw, img_info


def model_init(args: Namespace, cfg: Dict):
    """
    创建模型，赋值预训练权重
    """
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    anchors = torch.tensor(cfg['MODEL']['ANCHORS'])
    model = YOLOv2(anchors,
                   target_size=cfg['TEST']['IMGSIZE'],
                   num_classes=cfg['MODEL']['N_CLASSES'],
                   arch=cfg['MODEL']['BACKBONE'],
                   pretrained=cfg['MODEL']['BACKBONE_PRETRAINED']
                   ).to(device)

    assert args.ckpt, '--ckpt must be specified'
    if args.ckpt:
        print("=> loading checkpoint '{}'".format(args.ckpt))
        checkpoint = torch.load(args.ckpt, map_location=device)

        state_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model, device


def parse_info(outputs: List, info_img: List or Tuple):
    import random

    bboxes = list()
    confs = list()
    classes = list()
    colors = list()

    # x1/y1: 左上角坐标
    # x2/y2: 右下角坐标
    # conf: 置信度
    # cls_conf: 分类置信度
    # cls_pred: 分类下标
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs:
        cls_id = int(cls_pred)
        random.seed(cls_id)

        print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
        print('\t+ Label: %s, Conf: %.5f' % (VOCDataset.classes[cls_id], cls_conf.item()))
        y1, x1, y2, x2 = yolobox2label([y1, x1, y2, x2], info_img)
        bboxes.append([x1, y1, x2, y2])
        classes.append(cls_id)
        colors.append([random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)])
        confs.append(conf * cls_conf)

    return bboxes, confs, classes, colors


@torch.no_grad()
def process(input_data: Tensor, model: Module, device: torch.device,
            conf_thre=0.5, nms_thre=0.45, num_classes=20):
    # img: [1, 3, 416, 416]
    # 执行模型推理，批量计算每幅图像的预测框坐标以及对应的目标置信度+分类概率
    outputs = model(input_data.unsqueeze(0).to(device)).cpu()
    # outputs: [B, N_bbox, 4(xywh)+1(conf)+num_classes]
    # 图像后处理，执行预测边界框的坐标转换以及置信度阈值过滤+NMS IoU阈值过滤
    outputs = postprocess(outputs, num_classes, conf_thre=conf_thre, nms_thre=nms_thre)

    # [B, num_det, 7]
    return outputs


def show_bbox(save_dir: str,  # 保存路径
              img_raw_list: List[ndarray],  # 原始图像数据列表, BGR ndarray
              img_name_list: List[str],  # 图像名列表
              bboxes_list: List,  # 预测边界框
              confs_list: List,  # 预测边界框置信度
              names_list: List,  # 预测边界框对象名
              colors_list: List):  # 预测边界框绘制颜色
    """
    对于绘图，输入如下数据：
    1. 原始图像
    2. 预测框坐标
    3. 数据集名 + 分类概率
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_raw, img_name, bboxes, confs, names, colors in zip(
            img_raw_list, img_name_list, bboxes_list, confs_list, names_list, colors_list):
        im = img_raw

        for box, conf, pred_name, color in zip(bboxes, confs, names, colors):
            # box: [y1, x1, y2, x2]
            # print(box, name, color)
            assert len(box) == 4, box
            color = tuple([int(x) for x in color])

            # [y1, x1, y2, x2] -> [x1, y1] [x2, y2]
            p1, p2 = (int(box[1]), int(box[0])), (int(box[3]), int(box[2]))
            cv2.rectangle(im, p1, p2, color, 2)

            text_str = f'{pred_name} {conf:.3f}'
            w, h = cv2.getTextSize(text_str, 0, fontScale=0.5, thickness=1)[0]
            p1, p2 = (int(box[1]), int(box[0] - h)), (int(box[1] + w), int(box[0]))
            cv2.rectangle(im, p1, p2, color, thickness=-1)
            org = (int(box[1]), int(box[0]))
            cv2.putText(im, text_str, org, cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.5, color=(0, 0, 0), thickness=1)

        im_path = os.path.join(save_dir, img_name)
        print(f"\t+ img path: {im_path}")
        cv2.imwrite(im_path, im)


def main():
    """
    Visualize the detection result for the given image and the pre-trained model.
    操作流程：

    1. 解析命令行参数 + 配置文件
    2. 读取图像，预处理（图像通道转换 + 图像缩放 + 数据归一化 + 维度转换 + 数据格式转换）
    3. 创建模型，加载预训练权重
    4. 模型推理 + 数据后处理（置信度阈值过滤 + NMS阈值过滤）
    5. 预测框坐标转换
    6. 预测框绘制
    """
    args, cfg = parse_args()
    print("args:", args)

    print("=> Image Prerocess")
    input_data, img_raw, img_info = image_preprocess(args, cfg)
    print("=> Model Init")
    model, device = model_init(args, cfg)

    print("=> Process")
    conf_thre = cfg['TEST']['CONFTHRE']
    nms_thre = cfg['TEST']['NMSTHRE']
    if args.conf_thresh:
        conf_thre = args.conf_thresh
    if args.nms_thresh:
        nms_thre = args.nms_thresh
    num_classes = cfg['MODEL']['N_CLASSES']

    outputs = process(input_data, model, device, confthre=conf_thre, nms_thre=nms_thre, num_classes=num_classes)
    if outputs[0] is None:
        print("No Objects Deteted!!")
        return

    print("=> Parse INFO")
    bboxes, confs, pred_name_list, colors = parse_info(outputs[0], img_info[:6])

    img_raw_list = [img_raw]
    image_name = os.path.basename(args.image)
    img_name_list = [image_name]
    bboxes_list = [bboxes]
    confs_list = [confs]
    colors_list = [colors]

    save_dir = './results'
    show_bbox(save_dir, img_raw_list, img_name_list, bboxes_list, confs_list, pred_name_list, colors_list)


if __name__ == '__main__':
    main()
