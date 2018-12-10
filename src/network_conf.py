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
Date:2018-12-06 10:09:22
"""

from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Masking, Bidirectional, Dropout
from keras.models import Sequential
from keras.models import Model
from keras_contrib.layers import CRF

def create_model(maxlen, word_dict_size, label_size, is_train=False):
    """

    :param infer:
    :param maxlen:
    :param chars:
    :param word_size:
    :return:
    """
    sequence = Input(shape=(maxlen,), dtype='int32')
    embedded = Embedding(word_dict_size + 1, 32, input_length=maxlen, mask_zero=True)(sequence)
    blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
    output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
    model = Model(input=sequence, output=output)
    if is_train:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_model1(maxlen, word_dict_size, label_size, is_train=False):
    """

    :param infer:
    :param maxlen:
    :param chars:
    :param word_size:
    :return:
    """
    sequence = Input(shape=(maxlen,), dtype='int32')
    embedded = Embedding(word_dict_size + 1, 32, input_length=maxlen, mask_zero=True)(sequence)
    blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
    x = Dropout(0.2)(blstm)
    dense = TimeDistributed(Dense(5, activation='relu'))(x)

    crf = CRF(5, sparse_target=False, activation = "relu")
    crf_output = crf(dense)
    model = Model(input=sequence, output=crf_output)
    if is_train:
        model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    return model
