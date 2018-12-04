#!/usr/bin/env python
# coding: utf8
################################################################################
#
#
################################################################################
"""
Usage: 中文分词器基类
Date:2018-12-04 21:26:00
"""

import re

class Tokenizer(object):
    """接口
    """

    '''
    def cut(self, sentence, encoding = "utf8"):
        blocks = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', 
                sentence.decode("utf8").lower())
        toks = []
        for block in blocks:
            toks.extend(self._cut(block))

        return toks
    '''

    def cut(self, sentence):
        toks = self._cut(sentence)

        return toks

    def _cut(self, sentence):
        raise NotImplementedError
