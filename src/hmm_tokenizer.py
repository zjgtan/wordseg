#!/usr/bin/env python
# coding: utf8
################################################################################
#
# Copyright (c) 2016 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Usage: 
Authors: chenjiawei@baidu.com
Date:2018-12-04 23:13:48
"""

from tokenizer import Tokenizer

class HmmTokenizer(Tokenizer):
    """基于隐马尔科夫模型分词
    """
    def __init__(self):
        """
        hidden state: 
            B - begin 
            M - middle 
            E - end
            S - single
        """
        pass

    def fit(self, dataset):
        """训练
        """
        pass

    def _cut(self, sentence):
        """切词, Viterbi解码
        """
        pass



