# -*- coding: utf-8 -*-
from __future__ import absolute_import
from itertools import chain

import nltk

from textblob.utils import strip_punc
from textblob.base import BaseTokenizer
from textblob.decorators import requires_nltk_corpus


class WordTokenizer(BaseTokenizer):

    def tokenize(self, text, include_punc=True):
        tokens = nltk.tokenize.word_tokenize(text)
        if include_punc:
            return tokens
        else:
            return [word if word.startswith("'") else strip_punc(word, all=False)
                    for word in tokens if strip_punc(word, all=False)]


class SentenceTokenizer(BaseTokenizer):

    @requires_nltk_corpus
    def tokenize(self, text):
        return nltk.tokenize.sent_tokenize(text)


#: Convenience function for tokenizing sentences
sent_tokenize = SentenceTokenizer().itokenize

_word_tokenizer = WordTokenizer()  # Singleton word tokenizer
def word_tokenize(text, include_punc=True, *args, **kwargs):
    words = chain.from_iterable(
        _word_tokenizer.itokenize(sentence, include_punc=include_punc,
                                *args, **kwargs)
        for sentence in sent_tokenize(text))
    return words
