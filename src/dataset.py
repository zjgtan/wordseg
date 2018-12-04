#!/usr/bin/env python
# coding: utf8
################################################################################
#
#
################################################################################
"""
Usage: 
Date:2018-12-03 11:46:36
"""

import re


def load_test_set():
    """加载测试集
    """
    test_set = []
    for line in file("../data/icwb2-data/gold/pku_test_gold.utf8"):
        toks = re.split("\s+", line.rstrip().decode("utf8"))
        test_set.append(toks)

    return test_set

def load_train_set():
    test_set = []
    for line in file("../data/icwb2-data/training/pku_training.utf8"):
        toks = re.split("\s+", line.rstrip().decode("utf8"))
        test_set.append(toks)

    return test_set



