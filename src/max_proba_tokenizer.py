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
Date:2018-12-03 11:46:36
"""

import math
import logging
import json

from tokenizer import Tokenizer
from eval import Evaluation

class MaxProbaTokenizer(Tokenizer):
    """最大联合概率分词
    """

    def __init__(self, word_dict):
        """初始化
        Args:
            file_word_dict: word \t count
        """
        self.FREQ = word_dict
        self.V = len(word_dict)
        self.total_word = sum(self.FREQ.values())

    def get_word_proba(self, word):
        return math.log(self.FREQ.get(word, 0) + 1) - \
                math.log(self.total_word + self.V)

    def getDAG(self, sentence):
        """满足条件的分裂
        """
        # 用词典，较少存储量
        dag = {}
        N = len(sentence)
        for k in range(N, 0, -1):
            # 长度为K的子串如何切分
            temp_list = []
            for i in range(0, k):
                if sentence[i: k] in self.FREQ:
                    # 表示sentence[i:k]的子串在词典中
                    temp_list.append(i)

            if len(temp_list) == 0:
                temp_list.append(k - 1)

            dag[k] = temp_list

        return dag

    def calc(self, sentence, dag):
        """动态规划，计算最有路径
        target: max(p(w1, w2, ..., wn))
        """
        # 记录长度为k的子串的最佳分裂点和概率
        route = {}

        route[0] = (0, 0) # 边界值

        for k in range(1, len(sentence) + 1):
            # 长度为k的子串的最佳切分点
            scores = []
            for ix in dag[k]:
                # 当切分点为ix时，长度为ix的子串最佳切分的得分
                score = route[ix][1] + self.get_word_proba(sentence[ix:k])
                scores.append([ix, score])

            scores = sorted(scores, \
                    key = lambda (x, y): y, reverse = True)

            # 找出当前子串的最优分裂点
            route[k] = scores[0]

        return route

    def _cut(self, sentence):
        """切词
        """
        dag = self.getDAG(sentence)
        route = self.calc(sentence, dag)

        toks = []

        k = len(sentence)
        while k > 0:
            point, score = route[k]
            toks.append(sentence[point:k])
            k = point

        toks.reverse()

        return toks

if __name__ == "__main__":
    import dataset
    test_set = dataset.load_test_set()
    train_set = dataset.load_train_set()

    word_dict = {}
    for record in train_set:
        for tok in record:
            word_dict.setdefault(tok, 0)
            word_dict[tok] += 1

    tokenizer = MaxProbaTokenizer(word_dict)
    
    evaluator = Evaluation(test_set)
    precious, recall, F1 = evaluator.eval(tokenizer)
    print "precious: %f" % (precious)
    print "recall: %f" % (recall)
    print "F1: %f" % (F1)
