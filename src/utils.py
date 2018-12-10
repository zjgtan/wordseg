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

def update_counter(count_dict, key, count):
    count_dict.setdefault(key, 0)
    count_dict[key] += count


def get_bmes_label(toks):
    labels = []
    for i in range(len(toks)):
        for k in range(len(toks[i])):
            if k == 0:
                if len(toks[i]) == 1: 
                    labels.append("S")
                else:
                    labels.append("B")
            elif k == len(toks[i]) - 1:
                labels.append("E")
            else:
                labels.append("M")
    return labels


