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
import sys
reload(sys)
sys.setdefaultencoding("utf8")

import network_conf
import gzip
import logging
from keras.preprocessing.sequence import pad_sequences
import numpy as np


class EncoderDecoder(object):
    """数据集编解码
    """
    def __init__(self, labels):
        self.encoder = {}
        self.decoder = labels
        for ix, label in enumerate(labels):
            self.encoder[label] = ix

    def encode(self, key):
        if isinstance(key, list):
            return [self.encoder[e] for e in key]
        else:
            return self.encoder[key]

    def decode(self, key):
        if isinstance(key, list):
            return [self.decoder[e] for e in key]
        else:
            return self.decoder[key - 1]

    def decode_onehot(self, key):
        ix = 0
        for i in range(len(key)):
            if key == 1:
                ix = i

        return self.decode(ix - 1)

    def size(self):
        return len(self.encoder)


class Reader(object):
    """reader
    """
    def __init__(self, unigram_encoder_decoder,
            label_encoder_decoder, dataset, labels, maxlen=-1):
        self.unigram_encoder_decoder = unigram_encoder_decoder
        self.label_encoder_decoder = label_encoder_decoder
        self.dataset = dataset
        self.labels = labels

        self.maxlen = maxlen

    def read(self):
        for ix, record in enumerate(self.dataset):
            sentence = self.unigram_encoder_decoder.encode(list("".join(record)))
            sentence = [self.unigram_encoder_decoder.encode(word) + 1 
                    for word in list("".join(record))]
            if self.maxlen != -1:
                for _ in range(self.maxlen - len(sentence)):
                    sentence.append(0)
            target = self.label_encoder_decoder.encode(self.labels[ix])

            yield sentence, target

class LstmCrf(object):
    """分词模型
    """
    def __init__(self,
            unigram_encoder_decoder, label_encoder_decoder):
        self.unigram_encoder_decoder = unigram_encoder_decoder
        self.label_encoder_decoder = label_encoder_decoder

    def topology(self, job, maxlen):
        if job == "train":
            return network_conf.create_model1(
                    maxlen,
                    self.unigram_encoder_decoder.size(),
                    self.label_encoder_decoder.size(),
                    is_train = True)

        elif job == "predict":
            return network_conf.create_model(
                    maxlen,
                    self.unigram_encoder_decoder.size(),
                    self.label_encoder_decoder.size(),
                    is_train = True)

    def trans_dataset(self, dataset, maxlen):
        """
        """
        _dataset = []
        for ix, record in enumerate(dataset):
            record = list("".join(record))
            sentence = [self.unigram_encoder_decoder.encode(word) + 1
                    for word in record]

            if maxlen != -1:
                for _ in range(maxlen - len(sentence)):
                    sentence.append(0)
            _dataset.append(sentence)

        return _dataset

    def trans_labels(self, labels, maxlen):
        _labels = []
        for ix, record in enumerate(labels):
            sentence = [self.label_encoder_decoder.encode(word) + 1
                    for word in list("".join(record))]
            if maxlen != -1:
                for _ in range(maxlen - len(sentence)):
                    sentence.append(0)

            onehot_label = []
            for label in sentence:
                tmp = [0, 0, 0, 0, 0]
                tmp[label] = 1
                onehot_label.append(tmp)
            _labels.append(onehot_label)

        return _labels

    def fit(self, dataset, labels):
        """模型训练
        """
        maxlen = max([len(label) for label in labels])
        self.maxlen = maxlen
        #maxlen = 100

        dataset = self.trans_dataset(dataset, maxlen)
        labels = self.trans_labels(labels, maxlen)

        dataset = pad_sequences(dataset, maxlen)
        labels = pad_sequences(labels, maxlen)

        self.model = self.topology("train", maxlen)
        self.model.fit(dataset, labels, batch_size=128, nb_epoch=10) 

        score = self.model.evaluate(dataset, labels)
        print score

    def predict(self, sentence):
        """预测
        """
        # 对文本进行编码
        sentence = self.trans_dataset([sentence], self.maxlen)
        sentence = pad_sequences(sentence)
        # 执行预测
        results = self.model.predict(sentence)

        result = list(results)[0]
        result = np.argmax(result, axis = 1)
        print result
        # 输出bmse label
        labels = [self.label_encoder_decoder.decode(label) \
                for label in list(result)]

        return labels

if __name__ == "__main__":

    unigram_list = [u"我", u"爱", u"北", u"京", u"天", u"安", u"门"]
    labels = ["B", "M", "E", "S"]

    obj = LstmCrf(
            EncoderDecoder(unigram_list),
            EncoderDecoder(labels))

    dataset = [[u"我", u"爱", u"北京", u"天安门"] for _ in range(1000)]
    labels = [["S", "S", "B", "E", "B", "M", "E"] for _ in range(1000)]

    obj.fit(dataset, labels)
    print obj.predict(u"我爱北京天安门")





