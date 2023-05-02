# -*- coding: utf-8 -*-

"""
@date: 2023/5/2 下午4:31
@file: transform.py
@author: zj
@description: 
"""

import cv2
from yolo.data.transform import left_right_flip


def test_flip():
    src_img = cv2.imread("../assets/voc2007-test/000237.jpg")
    dst_img, bboxes = left_right_flip(src_img, [])

    cv2.imshow("src", src_img)
    cv2.imshow("dst", dst_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    test_flip()
