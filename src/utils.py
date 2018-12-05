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
