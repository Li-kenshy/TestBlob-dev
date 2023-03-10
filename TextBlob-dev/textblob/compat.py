# -*- coding: utf-8 -*-
import sys

PY2 = int(sys.version[0]) == 2

if PY2:
    from itertools import imap, izip
    import urllib2 as request
    from urllib import quote as urlquote
    from urllib import urlencode
    text_type = unicode
    binary_type = str
    string_types = (str, unicode)
    unicode = unicode
    basestring = basestring
    imap = imap
    izip = izip
    import unicodecsv as csv

    def implements_to_string(cls):
        cls.__unicode__ = cls.__str__
        cls.__str__ = lambda x: x.__unicode__().encode('utf-8')
        return cls
else:  # PY3
    from urllib import request
    from urllib.parse import quote as urlquote
    from urllib.parse import urlencode
    text_type = str
    binary_type = bytes
    string_types = (str,)
    unicode = str
    basestring = (str, bytes)
    imap = map
    izip = zip
    import csv

    implements_to_string = lambda x: x



def with_metaclass(meta, *bases):
    class metaclass(meta):  

        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    return type.__new__(metaclass, 'temporary_class', (), {})
