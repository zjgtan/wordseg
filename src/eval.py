#!/usr/bin/env python
# coding: utf8
################################################################################
#
#
################################################################################
"""
Usage: 
Date:2018-12-04 21:21:13
"""

class Metrics(object):
    """评估
    """
    def getPos(self, toks):
        """将切词结果转化为切词点的位置序列
        """
        positions = []

        for i in range(1, len(toks) + 1):
            positions.append(len(toks[:i]))

        return positions

    def precious(self, golds, preds):
        """准确率
        """
        self.total_pred_toks = 0
        self.correct_toks = 0

        for i in range(len(golds)):
            gold_positions = set(self.getPos(golds[i]))
            pred_positions = set(self.getPos(preds[i]))

            self.correct_toks += len(gold_positions & pred_positions)
            self.total_pred_toks += len(pred_positions)

        score = self.correct_toks * 1. / self.total_pred_toks

        return score

    def recall(self, golds, preds):
        """召回率
        """
        self.total_pred_toks = 0
        self.correct_toks = 0

        for i in range(len(golds)):
            gold_positions = set(self.getPos(golds[i]))
            pred_positions = set(self.getPos(preds[i]))

            self.correct_toks += len(gold_positions & pred_positions)
            self.total_pred_toks += len(gold_positions)

        score = self.correct_toks * 1. / self.total_pred_toks

        return score

    def F1(self, recall_score, precious_score):
        return 2 * (recall_score * precious_score) / (recall_score + precious_score)


class Evaluation(object):
    """模型评估
    """
    def __init__(self, golds):
        self.golds = golds

    def eval(self, tokenizer):
        """评估
        """
        preds = []
        for gold in self.golds:
            pred = tokenizer.cut("".join(gold))
            preds.append(pred)

        precious = Metrics().precious(self.golds, preds)
        recall = Metrics().recall(self.golds, preds)
        F1 = Metrics().F1(recall, precious)

        return precious, recall, F1
