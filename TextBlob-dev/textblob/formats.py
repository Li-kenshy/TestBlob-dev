# -*- coding: utf-8 -*-
from __future__ import absolute_import
import json
from collections import OrderedDict

from textblob.compat import PY2, csv
from textblob.utils import is_filelike

DEFAULT_ENCODING = 'utf-8'

class BaseFormat(object):
    def __init__(self, fp, **kwargs):
        pass

    def to_iterable(self):
        raise NotImplementedError('Must implement a "to_iterable" method.')

    @classmethod
    def detect(cls, stream):
        raise NotImplementedError('Must implement a "detect" class method.')

class DelimitedFormat(BaseFormat):

    delimiter = ","

    def __init__(self, fp, **kwargs):
        BaseFormat.__init__(self, fp, **kwargs)
        if PY2:
            reader = csv.reader(fp, delimiter=self.delimiter,
                                encoding=DEFAULT_ENCODING)
        else:
            reader = csv.reader(fp, delimiter=self.delimiter)
        self.data = [row for row in reader]

    def to_iterable(self):
        return self.data

    @classmethod
    def detect(cls, stream):
        try:
            csv.Sniffer().sniff(stream, delimiters=cls.delimiter)
            return True
        except (csv.Error, TypeError):
            return False


class CSV(DelimitedFormat):
    delimiter = ","


class TSV(DelimitedFormat):
    delimiter = "\t"


class JSON(BaseFormat):
    def __init__(self, fp, **kwargs):
        BaseFormat.__init__(self, fp, **kwargs)
        self.dict = json.load(fp)

    def to_iterable(self):
        return [(d['text'], d['label']) for d in self.dict]

    @classmethod
    def detect(cls, stream):
        try:
            json.loads(stream)
            return True
        except ValueError:
            return False


_registry = OrderedDict([
    ('csv', CSV),
    ('json', JSON),
    ('tsv', TSV),
])

def detect(fp, max_read=1024):
    if not is_filelike(fp):
        return None
    for Format in _registry.values():
        if Format.detect(fp.read(max_read)):
            fp.seek(0)
            return Format
        fp.seek(0)
    return None

def get_registry():
    return _registry

def register(name, format_class):
    get_registry()[name] = format_class
