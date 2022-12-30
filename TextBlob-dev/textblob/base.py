# -*- coding: utf-8 -*-
from __future__ import absolute_import
from abc import ABCMeta, abstractmethod

import nltk

from textblob.compat import with_metaclass


class BaseTagger(with_metaclass(ABCMeta)):

    @abstractmethod
    def tag(self, text, tokenize=True):

        return

##### NOUN PHRASE EXTRACTORS #####

class BaseNPExtractor(with_metaclass(ABCMeta)):

    @abstractmethod
    def extract(self, text):
        return

##### TOKENIZERS #####

class BaseTokenizer(with_metaclass(ABCMeta), nltk.tokenize.api.TokenizerI):
    @abstractmethod
    def tokenize(self, text):
        return

    def itokenize(self, text, *args, **kwargs):
        return (t for t in self.tokenize(text, *args, **kwargs))

##### SENTIMENT ANALYZERS ####


DISCRETE = 'ds'
CONTINUOUS = 'co'


class BaseSentimentAnalyzer(with_metaclass(ABCMeta)):
    kind = DISCRETE

    def __init__(self):
        self._trained = False

    def train(self):
        self._trained = True

    @abstractmethod
    def analyze(self, text):
        if not self._trained:
            self.train()
        return None

##### PARSERS #####

class BaseParser(with_metaclass(ABCMeta)):
    @abstractmethod
    def parse(self, text):
        return
