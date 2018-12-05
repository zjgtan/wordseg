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
import math

from tokenizer import Tokenizer
import utils

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
        self.state_trans_prob = {} # 状态转移矩阵
        self.state_emit_prob = {} # 发射转移矩阵
        self.start_state_prob = {} # 状态初始概率

    def get_labels(self, record):
        labels = []
        for i in range(len(record)):
            for k in range(len(record[i])):
                if k == 0:
                    if len(record[i]) == 1: 
                        labels.append("S")
                    else:
                        labels.append("B")
                elif k == len(record[i]) - 1:
                    labels.append("E")
                else:
                    labels.append("M")
        return labels

    def do_state_count(self, dataset):
        """统计
        """
        # 初始状态统计
        start_state_count = {}
        # 状态转移统计
        state_trans_count = {}
        # 状态发射统计
        state_emit_count = {}

        for record in dataset:
            labels = self.get_labels(record)
            record = "".join(record)

            for i in range(len(labels)):
                if i == 0:
                    # 更新初始状态统计
                    utils.update_counter(start_state_count,
                            labels[i], 1)
                    utils.update_counter(start_state_count,
                            "ALL", 1)
                else:
                    # 更新转移状态统计
                    state_trans_count.setdefault(labels[i - 1], {})
                    utils.update_counter(state_trans_count[labels[i - 1]],
                            labels[i], 1)
                    utils.update_counter(state_trans_count[labels[i - 1]],
                            "ALL", 1)

                # 更新状态发射统计
                state_emit_count.setdefault(labels[i], {})
                utils.update_counter(state_emit_count[labels[i]],
                        record[i], 1)
                utils.update_counter(state_emit_count[labels[i]],
                        "ALL", 1)

        return start_state_count, state_trans_count, state_emit_count

    def compute_proba(self, count_dict):
        """计算平滑的log prob
        """
        proba_dict = {}
        log_total = math.log(count_dict["ALL"] + len(count_dict) - 1)
        for key, value in count_dict.iteritems():
            if key == "ALL": continue
            proba_dict[key] = math.log(value + 1) - log_total

        proba_dict["NA"] = -1 * log_total

        return proba_dict

    def fit(self, dataset):
        """学习hmm的三个参数
        Args:
            dataset: list, 已切词数据集
        """
        start_state_count, state_trans_count, state_emit_count = \
                self.do_state_count(dataset)

        self.start_state_prob = self.compute_proba(start_state_count)

        self.state_trans_prob = {}
        for state, count_dict in state_trans_count.iteritems():
            self.state_trans_prob[state] = self.compute_proba(count_dict)

        self.state_emit_prob = {}
        for state, count_dict in state_emit_count.iteritems():
            self.state_emit_prob[state] = self.compute_proba(count_dict)


    def _cut(self, sentence):
        """切词, Viterbi解码
            遍历t=1...T上的所有路径的长度，取得最短路径
        """





if __name__ == "__main__":
    tokenizer = HmmTokenizer()
    tokenizer.fit([[u"我", u"爱", u"北京", u"天安门"]])
    print tokenizer.start_state_prob
    print tokenizer.state_trans_prob
    print tokenizer.state_emit_prob

