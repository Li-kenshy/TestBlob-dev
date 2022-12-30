# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sys
from textblob.compat import basestring, implements_to_string, PY2, binary_type


class ComparableMixin(object):


    def _compare(self, other, method):
        try:
            return method(self._cmpkey(), other._cmpkey())
        except (AttributeError, TypeError):
            # _cmpkey not implemented, or return different type,
            # so I can't compare with "other". Try the reverse comparison
            return NotImplemented

    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)


class BlobComparableMixin(ComparableMixin):


    def _compare(self, other, method):
        if isinstance(other, basestring):
            # Just compare with the other string
            return method(self._cmpkey(), other)
        return super(BlobComparableMixin, self)._compare(other, method)


@implements_to_string
class StringlikeMixin(object):


    def __repr__(self):
        class_name = self.__class__.__name__
        text = self.__unicode__().encode("utf-8") if PY2 else str(self)
        ret = '{cls}("{text}")'.format(cls=class_name,
                                        text=text)
        return binary_type(ret) if PY2 else ret

    def __str__(self):
        return self._strkey()

    def __len__(self):
        return len(self._strkey())

    def __iter__(self):
        return iter(self._strkey())

    def __contains__(self, sub):
        return sub in self._strkey()

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._strkey()[index]
        else:
            return self.__class__(self._strkey()[index])

    def find(self, sub, start=0, end=sys.maxsize):
        return self._strkey().find(sub, start, end)

    def rfind(self, sub, start=0, end=sys.maxsize):
        return self._strkey().rfind(sub, start, end)

    def index(self, sub, start=0, end=sys.maxsize):
        return self._strkey().index(sub, start, end)

    def rindex(self, sub, start=0, end=sys.maxsize):
        return self._strkey().rindex(sub, start, end)

    def startswith(self, prefix, start=0, end=sys.maxsize):
        return self._strkey().startswith(prefix, start, end)

    def endswith(self, suffix, start=0, end=sys.maxsize):
        return self._strkey().endswith(suffix, start, end)

    starts_with = startswith
    ends_with = endswith

    def title(self):
        return self.__class__(self._strkey().title())

    def format(self, *args, **kwargs):
        return self.__class__(self._strkey().format(*args, **kwargs))

    def split(self, sep=None, maxsplit=sys.maxsize):
        return self._strkey().split(sep, maxsplit)

    def strip(self, chars=None):
        return self.__class__(self._strkey().strip(chars))

    def upper(self):
        return self.__class__(self._strkey().upper())

    def lower(self):
        return self.__class__(self._strkey().lower())

    def join(self, iterable):
        return self.__class__(self._strkey().join(iterable))

    def replace(self, old, new, count=sys.maxsize):
        return self.__class__(self._strkey().replace(old, new, count))
