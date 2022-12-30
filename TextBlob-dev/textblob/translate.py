# -*- coding: utf-8 -*-
from __future__ import absolute_import

import codecs
import json
import re

from textblob.compat import PY2, request, urlencode
from textblob.exceptions import TranslatorError, NotTranslated


class Translator(object):


    url = "http://translate.google.com/translate_a/t?client=webapp&dt=bd&dt=ex&dt=ld&dt=md&dt=qca&dt=rw&dt=rm&dt=ss&dt=t&dt=at&ie=UTF-8&oe=UTF-8&otf=2&ssel=0&tsel=0&kc=1"

    headers = {
        'Accept': '*/*',
        'Connection': 'keep-alive',
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) '
            'AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.168 Safari/535.19')
    }

    def translate(self, source, from_lang='auto', to_lang='en', host=None, type_=None):
        if PY2:
            source = source.encode('utf-8')
        data = {"q": source}
        url = u'{url}&sl={from_lang}&tl={to_lang}&hl={to_lang}&tk={tk}&client={client}'.format(
            url=self.url,
            from_lang=from_lang,
            to_lang=to_lang,
            tk=_calculate_tk(source),
            client="te",
        )
        response = self._request(url, host=host, type_=type_, data=data)
        result = json.loads(response)
        if isinstance(result, list):
            try:
                result = result[0]
            except IndexError:
                pass
        self._validate_translation(source, result)
        return result

    def detect(self, source, host=None, type_=None):
        if PY2:
            source = source.encode('utf-8')
        if len(source) < 3:
            raise TranslatorError('Must provide a string with at least 3 characters.')
        data = {"q": source}
        url = u'{url}&sl=auto&tk={tk}&client={client}'.format(
            url=self.url,
            tk=_calculate_tk(source),
            client="te",
        )
        response = self._request(url, host=host, type_=type_, data=data)
        result, language = json.loads(response)
        return language

    def _validate_translation(self, source, result):
        if not result:
            raise NotTranslated('Translation API returned and empty response.')
        if PY2:
            result = result.encode('utf-8')
        if result.strip() == source.strip():
            raise NotTranslated('Translation API returned the input string unchanged.')

    def _request(self, url, host=None, type_=None, data=None):
        encoded_data = urlencode(data).encode('utf-8')
        req = request.Request(url=url, headers=self.headers, data=encoded_data)
        if host or type_:
            req.set_proxy(host=host, type=type_)
        resp = request.urlopen(req)
        content = resp.read()
        return content.decode('utf-8')


def _unescape(text):
    pattern = r'\\{1,2}u[0-9a-fA-F]{4}'
    return re.sub(pattern, lambda x: codecs.getdecoder('unicode_escape')(x.group())[0], text)


def _calculate_tk(source):

    def c_int(x, nbits=32):
        return (x & ((1 << (nbits - 1)) - 1)) - (x & (1 << (nbits - 1)))

    def c_uint(x, nbits=32):
        return x & ((1 << nbits) - 1)

    tkk = [406398, 561666268 + 1526272306]
    b = tkk[0]

    if PY2:
        d = map(ord, source)
    else:
        d = source.encode('utf-8')

    def RL(a, b):
        for c in range(0, len(b) - 2, 3):
            d = b[c + 2]
            d = ord(d) - 87 if d >= 'a' else int(d)
            xa = c_uint(a)
            d = xa >> d if b[c + 1] == '+' else xa << d
            a = a + d & 4294967295 if b[c] == '+' else a ^ d
        return c_int(a)

    a = b

    for di in d:
        a = RL(a + di, "+-a^+6")

    a = RL(a, "+-3^+b+-f")
    a ^= tkk[1]
    a = a if a >= 0 else ((a & 2147483647) + 2147483648)
    a %= pow(10, 6)

    tk = '{0:d}.{1:d}'.format(a, a ^ b)
    return tk
