# -*- coding: utf-8 -*-

"""
@date: 2023/4/23 上午9:55
@file: vocevaluator.py
@author: zj
@description:

每次输入的应该是

"""

from .evaluator import Evaluator


class VOCEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def put(self, outputs, img_info):
        super().put(outputs, img_info)

    def result(self):
        super().result()
