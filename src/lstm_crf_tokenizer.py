#!/usr/bin/env python
# coding: utf8
################################################################################
#
# Copyright (c) 2016 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Usage: 基于字粒度的lstm_crf训练分词模型
Authors: chenjiawei@baidu.com
Date:2018-12-05 12:47:28
"""

import utils

from lstm_crf import LstmCrf
from lstm_crf import EncoderDecoder

from tokenizer import Tokenizer



class LstmCrfTokenizer(Tokenizer):
    """lstm-crf模型
    """
    def get_unigrams(self, dataset):
        unigram_set = set()
        for record in dataset:
            for unigram in list("".join(record)):
                unigram_set.add(unigram)
        return list(unigram_set)

    def fit(self, dataset):
        """模型训练
        """
        # 转换数据集
        labels = [utils.get_bmes_label(record) for record in dataset]
        # 字节编码词典
        unigram_list = self.get_unigrams(dataset)

        unigram_encoder_decoder = EncoderDecoder(unigram_list)
        label_encoder_decoder = EncoderDecoder(["B", "M", "S", "E"])

        self.model = LstmCrf(unigram_encoder_decoder, 
                label_encoder_decoder)

        # 模型训练
        self.model.fit(dataset, labels)

    def _cut(self, sentence):
        """切词
        """
        best_route = self.model.predict(sentence)

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

    tokenizer = LstmCrfTokenizer()
    tokenizer.fit(train_set)
    tokenizer.init_predictor()
    
    evaluator = Evaluation(test_set)
    precious, recall, F1 = evaluator.eval(tokenizer)
    print "precious: %f" % (precious)
    print "recall: %f" % (recall)
    print "F1: %f" % (F1)

