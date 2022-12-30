# -*- coding: utf-8 -*-
import re
import string

PUNCTUATION_REGEX = re.compile('[{0}]'.format(re.escape(string.punctuation)))


def strip_punc(s, all=False):
    if all:
        return PUNCTUATION_REGEX.sub('', s.strip())
    else:
        return s.strip().strip(string.punctuation)


def lowerstrip(s, all=False):
    return strip_punc(s.lower().strip(), all=all)


def tree2str(tree, concat=' '):
    return concat.join([word for (word, tag) in tree])


def filter_insignificant(chunk, tag_suffixes=('DT', 'CC', 'PRP$', 'PRP')):
    good = []
    for word, tag in chunk:
        ok = True
        for suffix in tag_suffixes:
            if tag.endswith(suffix):
                ok = False
                break
        if ok:
            good.append((word, tag))
    return good


def is_filelike(obj):
    return hasattr(obj, 'read')
