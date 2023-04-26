# -*- coding: utf-8 -*-

"""
@date: 2023/4/26 下午2:13
@file: evaluator.py
@author: zj
@description: 
"""

from abc import ABC


class Evaluator(ABC):

    def put(self, outputs, img_info):
        pass

    def result(self):
        pass
