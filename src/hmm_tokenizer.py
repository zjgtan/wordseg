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

    def do_viterbi_decode(self, sentence):
        """切词, Viterbi解码
            遍历t=1...T上的所有路径的长度，取得最短路径
            p(w1, w2) = p(s1)p(w1|s1)p(s2|s1)p(w2|s1)
        """
        # 记录到达t时刻每个状态的最短路径长度，以及t-1时刻的出发状态
        route = {}
        steps = len(sentence)
        if steps == 0:
            return []

        for t in range(steps):
            route.setdefault(t, {})
            # 路径
            state_dict = {}
            for state in ["B", "E", "S", "M"]:
                # 遍历每个状态
                if t == 0: 
                    # 初始状态
                    score = self.start_state_prob.get(state, \
                            self.start_state_prob["NA"]) + \
                            self.state_emit_prob[state].get(sentence[t], \
                            self.state_emit_prob[state]["NA"])

                    last_state = "start"
                    route[t][state] = (score, last_state)
                else:
                    # 考虑上一步所有的状态
                    scores = []
                    for last_state in ["B", "E", "S", "M"]:
                        # 取上一步状态下的最优概率*一步转移概率*发射概率
                        score = route[t - 1][last_state][0] + \
                                self.state_trans_prob[last_state].get(state, 
                                        self.state_trans_prob[state]["NA"]) + \
                                self.state_emit_prob[state].get(sentence[t], \
                                self.state_emit_prob[state]["NA"])
                        scores.append((score, last_state))

                    score, last_state = sorted(scores, \
                            key=lambda (x, y): x, reverse = True)[0]

                    route[t][state] = (score, last_state)

        best_route = []
        best_last_state, (score, last_state) = \
                sorted(route[steps - 1].iteritems(), key = lambda (x, y): y[0],
                        reverse = True)[0]
        best_route.append(best_last_state)
        best_route.append(last_state)

        for t in range(steps - 2, 0, -1):
            score, last_state = route[t][last_state]
            best_route.append(last_state)

        best_route.reverse()

        return best_route

    def _cut(self, sentence):
        best_route = self.do_viterbi_decode(sentence)
        toks = []
        tok = ""
        for i in range(len(sentence)):
            tok += sentence[i]

            if best_route[i] in ["E", "S"]:
                toks.append(tok)
                tok = ""

        if tok != "":
            toks.append(tok)

        return toks

if __name__ == "__main__":
    import dataset
    from eval import Evaluation
    test_set = dataset.load_test_set()
    train_set = dataset.load_train_set()

    tokenizer = HmmTokenizer()
    tokenizer.fit(train_set)
    
    evaluator = Evaluation(test_set)
    precious, recall, F1 = evaluator.eval(tokenizer)
    print "precious: %f" % (precious)
    print "recall: %f" % (recall)
    print "F1: %f" % (F1)

