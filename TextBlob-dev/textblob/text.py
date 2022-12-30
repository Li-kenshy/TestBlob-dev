# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import string
import codecs
from itertools import chain
import types
import os
import re
from xml.etree import cElementTree

from .compat import text_type, basestring, imap, unicode, binary_type, PY2

try:
    MODULE = os.path.dirname(os.path.abspath(__file__))
except:
    MODULE = ""

SLASH, WORD, POS, CHUNK, PNP, REL, ANCHOR, LEMMA = \
    "&slash;", "word", "part-of-speech", "chunk", "preposition", "relation", "anchor", "lemma"

def decode_string(v, encoding="utf-8"):
    # 将给予的值作为按照统一字符编码标准编写的字符串返回
    if isinstance(encoding, basestring):
        encoding = ((encoding,),) + (("windows-1252",), ("utf-8", "ignore"))
    if isinstance(v, binary_type):
        for e in encoding:
            try:
                return v.decode(*e)
            except:
                pass
        return v
    return unicode(v)


def encode_string(v, encoding="utf-8"):
    #将给予的值作为python字节字符串返回
    if isinstance(encoding, basestring):
        encoding = ((encoding,),) + (("windows-1252",), ("utf-8", "ignore"))
    if isinstance(v, unicode):
        for e in encoding:
            try:
                return v.encode(*e)
            except:
                pass
        return v
    return str(v)

decode_utf8 = decode_string
encode_utf8 = encode_string


def isnumeric(strg):
    try:
        float(strg)
    except ValueError:
        return False
    return True



class lazydict(dict):

    def load(self):
        pass

    def _lazy(self, method, *args):
        """ 如果词典为空，调用lazydict.load()。
            将lazydict.method()替换为dict.method()并且调用它。
        """
        if dict.__len__(self) == 0:
            self.load()
            setattr(self, method, types.MethodType(getattr(dict, method), self))
        return getattr(dict, method)(self, *args)

    def __repr__(self):
        return self._lazy("__repr__")
    def __len__(self):
        return self._lazy("__len__")
    def __iter__(self):
        return self._lazy("__iter__")
    def __contains__(self, *args):
        return self._lazy("__contains__", *args)
    def __getitem__(self, *args):
        return self._lazy("__getitem__", *args)
    def __setitem__(self, *args):
        return self._lazy("__setitem__", *args)
    def setdefault(self, *args):
        return self._lazy("setdefault", *args)
    def get(self, *args, **kwargs):
        return self._lazy("get", *args)
    def items(self):
        return self._lazy("items")
    def keys(self):
        return self._lazy("keys")
    def values(self):
        return self._lazy("values")
    def update(self, *args):
        return self._lazy("update", *args)
    def pop(self, *args):
        return self._lazy("pop", *args)
    def popitem(self, *args):
        return self._lazy("popitem", *args)

class lazylist(list):

    def load(self):
        pass

    def _lazy(self, method, *args):
        """ 如果词典为空，调用lazylist.load()。
            将lazylist.method()替换为list.method()并且调用它。
        """
        if list.__len__(self) == 0:
            self.load()
            setattr(self, method, types.MethodType(getattr(list, method), self))
        return getattr(list, method)(self, *args)

    def __repr__(self):
        return self._lazy("__repr__")
    def __len__(self):
        return self._lazy("__len__")
    def __iter__(self):
        return self._lazy("__iter__")
    def __contains__(self, *args):
        return self._lazy("__contains__", *args)
    def insert(self, *args):
        return self._lazy("insert", *args)
    def append(self, *args):
        return self._lazy("append", *args)
    def extend(self, *args):
        return self._lazy("extend", *args)
    def remove(self, *args):
        return self._lazy("remove", *args)
    def pop(self, *args):
        return self._lazy("pop", *args)

UNIVERSAL = "universal"

NOUN, VERB, ADJ, ADV, PRON, DET, PREP, ADP, NUM, CONJ, INTJ, PRT, PUNC, X = \
    "NN", "VB", "JJ", "RB", "PR", "DT", "PP", "PP", "NO", "CJ", "UH", "PT", ".", "X"

def penntreebank2universal(token, tag):
    #返回具有简化的通用词性标记的（token, tag）元组。
    if tag.startswith(("NNP-", "NNPS-")):
        return (token, "%s-%s" % (NOUN, tag.split("-")[-1]))
    if tag in ("NN", "NNS", "NNP", "NNPS", "NP"):
        return (token, NOUN)
    if tag in ("MD", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
        return (token, VERB)
    if tag in ("JJ", "JJR", "JJS"):
        return (token, ADJ)
    if tag in ("RB", "RBR", "RBS", "WRB"):
        return (token, ADV)
    if tag in ("PRP", "PRP$", "WP", "WP$"):
        return (token, PRON)
    if tag in ("DT", "PDT", "WDT", "EX"):
        return (token, DET)
    if tag in ("IN",):
        return (token, PREP)
    if tag in ("CD",):
        return (token, NUM)
    if tag in ("CC",):
        return (token, CONJ)
    if tag in ("UH",):
        return (token, INTJ)
    if tag in ("POS", "RP", "TO"):
        return (token, PRT)
    if tag in ("SYM", "LS", ".", "!", "?", ",", ":", "(", ")", "\"", "#", "$"):
        return (token, PUNC)
    return (token, X)

TOKEN = re.compile(r"(\S+)\s")

#处理普通标点符号
PUNCTUATION = \
punctuation = ".,;:!?()[]{}`''\"@#$^&*+-|=~_"

#处理普通缩写词
ABBREVIATIONS = abbreviations = set((
    "a.", "adj.", "adv.", "al.", "a.m.", "c.", "cf.", "comp.", "conf.", "def.",
    "ed.", "e.g.", "esp.", "etc.", "ex.", "f.", "fig.", "gen.", "id.", "i.e.",
    "int.", "l.", "m.", "Med.", "Mil.", "Mr.", "n.", "n.q.", "orig.", "pl.",
    "pred.", "pres.", "p.m.", "ref.", "v.", "vs.", "w/"
))

# 单独字母
RE_ABBR1 = re.compile("^[A-Za-z]\.$") 
#可替换字母
RE_ABBR2 = re.compile("^([A-Za-z]\.)+$")    
#辅音字母之前的大写字母
RE_ABBR3 = re.compile("^[A-Z][" + "|".join(
        "bcdfghjklmnpqrstvwxz") + "]+.$")

# 处理文字颜图像
EMOTICONS = {
    ("love" , +1.00): set(("<3", "♥")),
    ("grin" , +1.00): set((">:D", ":-D", ":D", "=-D", "=D", "X-D", "x-D", "XD", "xD", "8-D")),
    ("taunt", +0.75): set((">:P", ":-P", ":P", ":-p", ":p", ":-b", ":b", ":c)", ":o)", ":^)")),
    ("smile", +0.50): set((">:)", ":-)", ":)", "=)", "=]", ":]", ":}", ":>", ":3", "8)", "8-)")),
    ("wink" , +0.25): set((">;]", ";-)", ";)", ";-]", ";]", ";D", ";^)", "*-)", "*)")),
    ("gasp" , +0.05): set((">:o", ":-O", ":O", ":o", ":-o", "o_O", "o.O", "°O°", "°o°")),
    ("worry", -0.25): set((">:/",  ":-/", ":/", ":\\", ">:\\", ":-.", ":-s", ":s", ":S", ":-S", ">.>")),
    ("frown", -0.75): set((">:[", ":-(", ":(", "=(", ":-[", ":[", ":{", ":-<", ":c", ":-c", "=/")),
    ("cry"  , -1.00): set((":'(", ":'''(", ";'("))
}

RE_EMOTICONS = [r" ?".join([re.escape(each) for each in e]) for v in EMOTICONS.values() for e in v]
RE_EMOTICONS = re.compile(r"(%s)($|\s)" % "|".join(RE_EMOTICONS))
RE_SARCASM = re.compile(r"\( ?\! ?\)")

# 处理后缀缩略字
replacements = {
     "'d": " 'd",
     "'m": " 'm",
     "'s": " 's",
    "'ll": " 'll",
    "'re": " 're",
    "'ve": " 've",
    "n't": " n't"
}

# 处理段落空行 (在句子结尾标记\n\n )。
EOS = "END-OF-SENTENCE"

def find_tokens(string, punctuation=PUNCTUATION, abbreviations=ABBREVIATIONS, replace=replacements, linebreak=r"\n{2,}"):
    #返回句子列表。每个句子都是一个以空格分隔的标记（单词）字符串
    #处理缩写的常见情况。 
    #标点符号与其他单词分开。句号（或 ？！）标记句子的结尾。
    #没有结束句点的标题可以由换行符推断出

    punctuation = tuple(punctuation.replace(".", ""))
    for a, b in list(replace.items()):
        string = re.sub(a, b, string)
    if isinstance(string, unicode):
        string = unicode(string).replace("“", " “ ")\
                                .replace("”", " ” ")\
                                .replace("‘", " ‘ ")\
                                .replace("’", " ’ ")\
                                .replace("'", " ' ")\
                                .replace('"', ' " ')
    # 折叠空格
    string = re.sub("\r\n", "\n", string)
    string = re.sub(linebreak, " %s " % EOS, string)
    string = re.sub(r"\s+", " ", string)
    tokens = []
    for t in TOKEN.findall(string+" "):
        if len(t) > 0:
            tail = []
            while t.startswith(punctuation) and \
              not t in replace:
                # 拆分前导标点符号
                if t.startswith(punctuation):
                    tokens.append(t[0]); t=t[1:]
            while t.endswith(punctuation+(".",)) and \
              not t in replace:
                # 拆分尾随标点符号
                if t.endswith(punctuation):
                    tail.append(t[-1]); t=t[:-1]
                # 拆分一个splitting周期前的省略号
                if t.endswith("..."):
                    tail.append("..."); t=t[:-3].rstrip(".")
                # 拆分周期
                if t.endswith("."):
                    if t in abbreviations or \
                      RE_ABBR1.match(t) is not None or \
                      RE_ABBR2.match(t) is not None or \
                      RE_ABBR3.match(t) is not None:
                        break
                    else:
                        tail.append(t[-1]); t=t[:-1]
            if t != "":
                tokens.append(t)
            tokens.extend(reversed(tail))
    sentences, i, j = [[]], 0, 0
    while j < len(tokens):
        if tokens[j] in ("...", ".", "!", "?", EOS):
            # 处理引文、尾随括号、重复标点符号
            while j < len(tokens) \
                    and tokens[j] in ("'", "\"", u"”", u"’", "...", ".", "!", "?", ")", EOS):
                if tokens[j] in ("'", "\"") and sentences[-1].count(tokens[j]) % 2 == 0:
                    break
                j += 1
            sentences[-1].extend(t for t in tokens[i:j] if t != EOS)
            sentences.append([])
            i = j
        j += 1
    sentences[-1].extend(tokens[i:j])
    sentences = (" ".join(s) for s in sentences if len(s) > 0)
    sentences = (RE_SARCASM.sub("(!)", s) for s in sentences)
    sentences = [RE_EMOTICONS.sub(
        lambda m: m.group(1).replace(" ", "") + m.group(2), s) for s in sentences]
    return sentences

# Pattern的文本解析器基于Brill的算法。
# Brill的算法会自动获取已知单词的词典，以及一组用于从训练语料库中标记未知单词的规则。
# 词典规则用于根据单词形态（前缀、后缀等）标记未知单词。 
# 上下文规则用于根据单词在句子中的角色标记所有单词。
# 命名实体规则用于发现专有名词 （NNP）。

def _read(path, encoding="utf-8", comment=";;;"):
    """ 在给定路径处返回文件中各行的迭代器，
        注释并将每一行代码解码为Unicode
    """
    if path:
        if isinstance(path, basestring) and os.path.exists(path):
            #通过文件路径
            if PY2:
                f = codecs.open(path, 'r', encoding='utf-8')
            else:
                f = open(path, 'r', encoding='utf-8')
        elif isinstance(path, basestring):
            #通过字符串
            f = path.splitlines()
        elif hasattr(path, "read"):
            #通过字符串缓冲器
            f = path.read().splitlines()
        else:
            f = path
        for i, line in enumerate(f):
            line = line.strip(codecs.BOM_UTF8) if i == 0 and isinstance(line, binary_type) else line
            line = line.strip()
            line = decode_utf8(line)
            if not line or (comment and line.startswith(comment)):
                continue
            yield line
    return


class Lexicon(lazydict):

    def __init__(self, path="", morphology=None, context=None, entities=None, NNP="NNP", language=None):
        """一个收录单词及其词性标签的字典
           对于未知单词，可以使用单词形态、上下文和命名实体的规则进行推理。
        """
        self._path = path
        self._language  = language
        self.morphology = Morphology(self, path=morphology)
        self.context    = Context(self, path=context)
        self.entities   = Entities(self, path=entities, tag=NNP)

    def load(self):
        dict.update(self, (x.split(" ")[:2] for x in _read(self._path) if x.strip()))

    @property
    def path(self):
        return self._path

    @property
    def language(self):
        return self._language

class Rules:

    def __init__(self, lexicon={}, cmd={}):
        self.lexicon, self.cmd = lexicon, cmd

    def apply(self, x):
       #将规则应用于给定的字词或字词列表
        return x

class Morphology(lazylist, Rules):

    def __init__(self, lexicon={}, path=""):
        #基于单词形态（前缀、后缀）的规则列表。
        cmd = ("char", 
            "haspref", 
             "hassuf",
            "addpref",
             "addsuf",
         "deletepref",
          "deletesuf",
           "goodleft",
          "goodright",
        )
        cmd = dict.fromkeys(cmd, True)
        cmd.update(("f" + k, v) for k, v in list(cmd.items()))
        Rules.__init__(self, lexicon, cmd)
        self._path = path

    @property
    def path(self):
        return self._path

    def load(self):
        list.extend(self, (x.split() for x in _read(self._path)))

    def apply(self, token, previous=(None, None), next=(None, None)):
        #将词法规则应用于给定的标记，即 [word, tag] 列表。
        w = token[0]
        for r in self:
            if r[1] in self.cmd: 
                f, x, pos, cmd = bool(0), r[0], r[-2], r[1].lower()
            if r[2] in self.cmd:
                f, x, pos, cmd = bool(1), r[1], r[-2], r[2].lower().lstrip("f")
            if f and token[1] != r[0]:
                continue
            if (cmd == "char"       and x in w) \
            or (cmd == "haspref"    and w.startswith(x)) \
            or (cmd == "hassuf"     and w.endswith(x)) \
            or (cmd == "addpref"    and x + w in self.lexicon) \
            or (cmd == "addsuf"     and w + x in self.lexicon) \
            or (cmd == "deletepref" and w.startswith(x) and w[len(x):] in self.lexicon) \
            or (cmd == "deletesuf"  and w.endswith(x) and w[:-len(x)] in self.lexicon) \
            or (cmd == "goodleft"   and x == next[0]) \
            or (cmd == "goodright"  and x == previous[0]):
                token[1] = pos
        return token

    def insert(self, i, tag, affix, cmd="hassuf", tagged=None):
        """ 插入一个新规则，该规则将给定标签分配给具有给定词缀的单词， 
            例如, Morphology.append("RB", "-ly").
        """
        if affix.startswith("-") and affix.endswith("-"):
            affix, cmd = affix[+1:-1], "char"
        if affix.startswith("-"):
            affix, cmd = affix[+1:-0], "hassuf"
        if affix.endswith("-"):
            affix, cmd = affix[+0:-1], "haspref"
        if tagged:
            r = [tagged, affix, "f"+cmd.lstrip("f"), tag, "x"]
        else:
            r = [affix, cmd.lstrip("f"), tag, "x"]
        lazylist.insert(self, i, r)

    def append(self, *args, **kwargs):
        self.insert(len(self)-1, *args, **kwargs)

    def extend(self, rules=[]):
        for r in rules:
            self.append(*r)

# Brill 算法按以下格式生成Context Rules： 
# VBD VB PREVTAG TO => unknown word tagged VBD changes to VB if preceded by a word tagged TO.

class Context(lazylist, Rules):

    def __init__(self, lexicon={}, path=""):
       #基于上下文（前后单词）的规则列表。

        cmd = ("prevtag", 
               "nexttag", 
              "prev2tag", 
              "next2tag", 
           "prev1or2tag", 
           "next1or2tag", 
        "prev1or2or3tag", 
        "next1or2or3tag", 
           "surroundtag", 
                 "curwd", 
                "prevwd", 
                "nextwd", 
            "prev1or2wd", 
            "next1or2wd", 
         "next1or2or3wd", 
         "prev1or2or3wd", 
             "prevwdtag", 
             "nextwdtag", 
             "wdprevtag", 
             "wdnexttag", 
             "wdand2aft", 
          "wdand2tagbfr", 
          "wdand2tagaft", 
               "lbigram", 
               "rbigram", 
            "prevbigram", 
            "nextbigram", 
        )
        Rules.__init__(self, lexicon, dict.fromkeys(cmd, True))
        self._path = path

    @property
    def path(self):
        return self._path

    def load(self):
        # ["VBD", "VB", "PREVTAG", "TO"]
        list.extend(self, (x.split() for x in _read(self._path)))

    def apply(self, tokens):
       #将上下文规则应用于给定的标记列表，其中每个标记都是一个 [word,tag] 列表。

        o = [("STAART", "STAART")] * 3
        t = o + tokens + o
        for i, token in enumerate(t):
            for r in self:
                if token[1] == "STAART":
                    continue
                if token[1] != r[0] and r[0] != "*":
                    continue
                cmd, x, y = r[2], r[3], r[4] if len(r) > 4 else ""
                cmd = cmd.lower()
                if (cmd == "prevtag"        and x ==  t[i-1][1]) \
                or (cmd == "nexttag"        and x ==  t[i+1][1]) \
                or (cmd == "prev2tag"       and x ==  t[i-2][1]) \
                or (cmd == "next2tag"       and x ==  t[i+2][1]) \
                or (cmd == "prev1or2tag"    and x in (t[i-1][1], t[i-2][1])) \
                or (cmd == "next1or2tag"    and x in (t[i+1][1], t[i+2][1])) \
                or (cmd == "prev1or2or3tag" and x in (t[i-1][1], t[i-2][1], t[i-3][1])) \
                or (cmd == "next1or2or3tag" and x in (t[i+1][1], t[i+2][1], t[i+3][1])) \
                or (cmd == "surroundtag"    and x ==  t[i-1][1] and y == t[i+1][1]) \
                or (cmd == "curwd"          and x ==  t[i+0][0]) \
                or (cmd == "prevwd"         and x ==  t[i-1][0]) \
                or (cmd == "nextwd"         and x ==  t[i+1][0]) \
                or (cmd == "prev1or2wd"     and x in (t[i-1][0], t[i-2][0])) \
                or (cmd == "next1or2wd"     and x in (t[i+1][0], t[i+2][0])) \
                or (cmd == "prevwdtag"      and x ==  t[i-1][0] and y == t[i-1][1]) \
                or (cmd == "nextwdtag"      and x ==  t[i+1][0] and y == t[i+1][1]) \
                or (cmd == "wdprevtag"      and x ==  t[i-1][1] and y == t[i+0][0]) \
                or (cmd == "wdnexttag"      and x ==  t[i+0][0] and y == t[i+1][1]) \
                or (cmd == "wdand2aft"      and x ==  t[i+0][0] and y == t[i+2][0]) \
                or (cmd == "wdand2tagbfr"   and x ==  t[i-2][1] and y == t[i+0][0]) \
                or (cmd == "wdand2tagaft"   and x ==  t[i+0][0] and y == t[i+2][1]) \
                or (cmd == "lbigram"        and x ==  t[i-1][0] and y == t[i+0][0]) \
                or (cmd == "rbigram"        and x ==  t[i+0][0] and y == t[i+1][0]) \
                or (cmd == "prevbigram"     and x ==  t[i-2][1] and y == t[i-1][1]) \
                or (cmd == "nextbigram"     and x ==  t[i+1][1] and y == t[i+2][1]):
                    t[i] = [t[i][0], r[1]]
        return t[len(o):-len(o)]

    def insert(self, i, tag1, tag2, cmd="prevtag", x=None, y=None):
        """ 插入一条新规则，将带有标签 1 的单词更新为标签 2， 
            给定约束 x 和 y, 例如：Context.append("TO < NN", "VB")
        """
        if " < " in tag1 and not x and not y:
            tag1, x = tag1.split(" < "); cmd="prevtag"
        if " > " in tag1 and not x and not y:
            x, tag1 = tag1.split(" > "); cmd="nexttag"
        lazylist.insert(self, i, [tag1, tag2, cmd, x or "", y or ""])

    def append(self, *args, **kwargs):
        self.insert(len(self)-1, *args, **kwargs)

    def extend(self, rules=[]):
        for r in rules:
            self.append(*r)

RE_ENTITY1 = re.compile(r"^http://")                            
RE_ENTITY2 = re.compile(r"^www\..*?\.[com|org|net|edu|de|uk]$") 
RE_ENTITY3 = re.compile(r"^[\w\-\.\+]+@(\w[\w\-]+\.)+[\w\-]+$") 

class Entities(lazydict, Rules):

    def __init__(self, lexicon={}, path="", tag="NNP"):
        #收录命名实体及其标签的字典。使用正则表达式表示域名和电子邮件地址。
        cmd = (
            "pers",
             "loc", 
             "org",
        )
        Rules.__init__(self, lexicon, cmd)
        self._path = path
        self.tag   = tag

    @property
    def path(self):
        return self._path

    def load(self):
        for x in _read(self.path):
            x = [x.lower() for x in x.split()]
            dict.setdefault(self, x[0], []).append(x)

    def apply(self, tokens):
       #将命名实体识别器应用于给定的token，其中每个标记都是一个 [word,tag] 列表。

        i = 0
        while i < len(tokens):
            w = tokens[i][0].lower()
            if RE_ENTITY1.match(w) \
            or RE_ENTITY2.match(w) \
            or RE_ENTITY3.match(w):
                tokens[i][1] = self.tag
            if w in self:
                for e in self[w]:
                    # 查看连续的单词是否与命名实体匹配。
                    e, tag = (e[:-1], "-"+e[-1].upper()) if e[-1] in self.cmd else (e, "")
                    b = True
                    for j, e in enumerate(e):
                        if i + j >= len(tokens) or tokens[i+j][0].lower() != e:
                            b = False; break
                    if b:
                        for token in tokens[i:i+j+1]:
                            token[1] = (token[1] == "NNPS" and token[1] or self.tag) + tag
                        i += j
                        break
            i += 1
        return tokens

    def append(self, entity, name="pers"):
        #给lexicon附上一个命名实体
        e = [s.lower() for s in entity.split(" ") + [name]]
        self.setdefault(e[0], []).append(e)

    def extend(self, entities):
        for entity, name in entities:
            self.append(entity, name)


# 情感标签
MOOD  = "mood"  
IRONY = "irony" 

NOUN, VERB, ADJECTIVE, ADVERB = \
    "NN", "VB", "JJ", "RB"

RE_SYNSET = re.compile(r"^[acdnrv][-_][0-9]+$")

def avg(list):
    return sum(list) / float(len(list) or 1)

class Score(tuple):

    def __new__(self, polarity, subjectivity, assessments=[]):
        return tuple.__new__(self, [polarity, subjectivity])

    def __init__(self, polarity, subjectivity, assessments=[]):
        self.assessments = assessments

class Sentiment(lazydict):

    def __init__(self, path="", language=None, synset=None, confidence=None, **kwargs):
        #一个收录形容词和对立性分数（polarity score）的词典
        #每个单词 POS 标记的值是一个元组，其值为对立性 （-1.0-1.0）、主观性 （0.0-1.0） 和强度 （0.5-2.0）。
        self._path       = path  
        self._language   = None   
        self._confidence = None   
        self._synset     = synset 
        self._synsets    = {}     
        self.labeler     = {}    
        self.tokenizer   = kwargs.get("tokenizer", find_tokens)
        self.negations   = kwargs.get("negations", ("no", "not", "n't", "never"))
        self.modifiers   = kwargs.get("modifiers", ("RB",))
        self.modifier    = kwargs.get("modifier" , lambda w: w.endswith("ly"))

    @property
    def path(self):
        return self._path

    @property
    def language(self):
        return self._language

    @property
    def confidence(self):
        return self._confidence

    def load(self, path=None):
        #从给定路径加载XML文件（带有情绪注释）。默认情况下，情绪路径是延迟加载的。

        if not path:
            path = self._path
        if not os.path.exists(path):
            return
        words, synsets, labels = {}, {}, {}
        xml = cElementTree.parse(path)
        xml = xml.getroot()
        for w in xml.findall("word"):
            if self._confidence is None \
            or self._confidence <= float(w.attrib.get("confidence", 0.0)):
                w, pos, p, s, i, label, synset = (
                    w.attrib.get("form"),
                    w.attrib.get("pos"),
                    w.attrib.get("polarity", 0.0),
                    w.attrib.get("subjectivity", 0.0),
                    w.attrib.get("intensity", 1.0),
                    w.attrib.get("label"),
                    w.attrib.get(self._synset)
                )
                psi = (float(p), float(s), float(i))
                if w:
                    words.setdefault(w, {}).setdefault(pos, []).append(psi)
                if w and label:
                    labels[w] = label
                if synset:
                    synsets.setdefault(synset, []).append(psi)
        self._language = xml.attrib.get("language", self._language)
        # 每个词性标签的所有词义的平均分数。
        for w in words:
            words[w] = dict((pos, [avg(each) for each in zip(*psi)]) for pos, psi in words[w].items())
        # 所有词性标签的平均分数。
        for w, pos in list(words.items()):
            words[w][None] = [avg(each) for each in zip(*pos.values())]
        # 每个同义词集中所有同义词的平均分数。
        for id, psi in synsets.items():
            synsets[id] = [avg(each) for each in zip(*psi)]
        dict.update(self, words)
        dict.update(self.labeler, labels)
        dict.update(self._synsets, synsets)

    def synset(self, id, pos=ADJECTIVE):
        """ 返回一个给定同义词集id的 （polarity, subjectivity）元组。
            例如，形容词“horrible”在WordNet中的id为193480，那么会返回：
            Sentiment.synset(193480, pos="JJ") => (-0.6, 1.0, 1.0).
        """
        id = str(id).zfill(8)
        if not id.startswith(("n-", "v-", "a-", "r-")):
            if pos == NOUN:
                id = "n-" + id
            if pos == VERB:
                id = "v-" + id
            if pos == ADJECTIVE:
                id = "a-" + id
            if pos == ADVERB:
                id = "r-" + id
        if dict.__len__(self) == 0:
            self.load()
        return tuple(self._synsets.get(id, (0.0, 0.0))[:2])

    def __call__(self, s, negation=True, **kwargs):
        """ 返回一个给定句子（polarity, subjectivity）的元组。
            对立性于-1.0和1.0之间，主观性介于0.0和1.0之间。句子可以是字符串、同义词集、文本、句子、块、单词。
            可以给出一个可选的权重参数，为接受单词列表并返回权重的函数。
        """
        def avg(assessments, weighted=lambda w: 1):
            s, n = 0, 0
            for words, score in assessments:
                w = weighted(words)
                s += w * score
                n += w
            return s / float(n or 1)
        if hasattr(s, "gloss"):
            a = [(s.synonyms[0],) + self.synset(s.id, pos=s.pos) + (None,)]

        elif isinstance(s, basestring) and RE_SYNSET.match(s) and hasattr(s, "synonyms"):
            a = [(s.synonyms[0],) + self.synset(s.id, pos=s.pos) + (None,)]

        elif isinstance(s, basestring):
            a = self.assessments(((w.lower(), None) for w in " ".join(self.tokenizer(s)).split()), negation)

        elif hasattr(s, "sentences"):
            a = self.assessments(((w.lemma or w.string.lower(), w.pos[:2])
                                  for w in chain.from_iterable(s)), negation)

        elif hasattr(s, "lemmata"):
            a = self.assessments(((w.lemma or w.string.lower(), w.pos[:2]) for w in s.words), negation)

        elif hasattr(s, "lemma"):
            a = self.assessments(((s.lemma or s.string.lower(), s.pos[:2]),), negation)

        elif hasattr(s, "terms"):
            a = self.assessments(chain.from_iterable(((w, None), (None, None)) for w in s), negation)
            kwargs.setdefault("weight", lambda w: s.terms[w[0]])

        elif isinstance(s, dict):
            a = self.assessments(chain.from_iterable(((w, None), (None, None)) for w in s), negation)
            kwargs.setdefault("weight", lambda w: s[w[0]])

        elif isinstance(s, list):
            a = self.assessments(((w, None) for w in s), negation)
        else:
            a = []
        weight = kwargs.get("weight", lambda w: 1) 
        return Score(polarity = avg( [(w, p) for w, p, s, x in a], weight ),
                 subjectivity = avg([(w, s) for w, p, s, x in a], weight),
                  assessments = a)

    def assessments(self, words=[], negation=True):
        """ 返回给定单词列表的（chunk, polarity, subjectivity, label）元组列表：
            其中 chunk 是连续单词的列表：已知单词选项前面有一个修饰符（“very good”）或否定符（“not”）。
        """
        a = []
        m = None 
        n = None 
        for w, pos in words:
            # 通过词性标签评估已知单词。
            # 如果包含未知单词(对立性为0.0并且主观性为0.0)，会降低平均值。
            if w is None:
                continue
            if w in self and pos in self[w]:
                p, s, i = self[w][pos]

                if m is None:
                    a.append(dict(w=[w], p=p, s=s, i=i, n=1, x=self.labeler.get(w)))
                
                if m is not None:
                    a[-1]["w"].append(w)
                    a[-1]["p"] = max(-1.0, min(p * a[-1]["i"], +1.0))
                    a[-1]["s"] = max(-1.0, min(s * a[-1]["i"], +1.0))
                    a[-1]["i"] = i
                    a[-1]["x"] = self.labeler.get(w)

                if n is not None:
                    a[-1]["w"].insert(0, n)
                    a[-1]["i"] = 1.0 / a[-1]["i"]
                    a[-1]["n"] = -1

                m = None
                n = None
                if pos and pos in self.modifiers or any(map(self[w].__contains__, self.modifiers)):
                    m = (w, pos)
                if negation and w in self.negations:
                    n = w
            else:

                if negation and w in self.negations:
                    n = w

                elif n and len(w.strip("'")) > 1:
                    n = None

                if n is not None and m is not None and (pos in self.modifiers or self.modifier(m[0])):
                    a[-1]["w"].append(n)
                    a[-1]["n"] = -1
                    n = None

                elif m and len(w) > 2:
                    m = None
                # 感叹号增强前一个单词的得分评估。
                if w == "!" and len(a) > 0:
                    a[-1]["w"].append("!")
                    a[-1]["p"] = max(-1.0, min(a[-1]["p"] * 1.25, +1.0))

                if w == "(!)":
                    a.append(dict(w=[w], p=0.0, s=1.0, i=1.0, n=1, x=IRONY))

                if w.isalpha() is False and len(w) <= 5 and w not in PUNCTUATION: 
                    for (type, p), e in EMOTICONS.items():
                        if w in imap(lambda e: e.lower(), e):
                            a.append(dict(w=[w], p=p, s=1.0, i=1.0, n=1, x=MOOD))
                            break
        for i in range(len(a)):
            w = a[i]["w"]
            p = a[i]["p"]
            s = a[i]["s"]
            n = a[i]["n"]
            x = a[i]["x"]

            a[i] = (w, p * -0.5 if n < 0 else p, s, x)
        return a

    def annotate(self, word, pos=None, polarity=0.0, subjectivity=0.0, intensity=1.0, label=None):
        #用对立性、主观性和强度分数注释给定的单词，以及可选的语义标签。

        w = self.setdefault(word, {})
        w[pos] = w[None] = (polarity, subjectivity, intensity)
        if label:
            self.labeler[word] = label

# 如果未知单词仅包含数字和-,.:/%$，则将其识别为数字。
CD = re.compile(r"^[0-9\-\,\.\:\/\%\$]+$")

def _suffix_rules(token, tag="NN"):
    #基于单词后缀的英语默认形态标记规则。

    if isinstance(token, (list, tuple)):
        token, tag = token
    if token.endswith("ing"):
        tag = "VBG"
    if token.endswith("ly"):
        tag = "RB"
    if token.endswith("s") and not token.endswith(("is", "ous", "ss")):
        tag = "NNS"
    if token.endswith(("able", "al", "ful", "ible", "ient", "ish", "ive", "less", "tic", "ous")) or "-" in token:
        tag = "JJ"
    if token.endswith("ed"):
        tag = "VBN"
    if token.endswith(("ate", "ify", "ise", "ize")):
        tag = "VBP"
    return [token, tag]

def find_tags(tokens, lexicon={}, model=None, morphology=None, context=None, entities=None, default=("NN", "NNP", "CD"), language="en", map=None, **kwargs):
    """
        单词使用给定的（word, tag）lexicon进行标记。
        默认情况下，未知单词标记为NN。
        以大写字母开头的未知单词标记为NNP。
        仅由数字和标点符号组成的未知单词标记为CD。
        然后使用morphological rules改进未知单词的评估。
        所有单词的评估都会通过contextual rules进行改进。
        如果给出了模型，则使用给出的模型对未知单词进行评估改进，而不是morphological or contextual rules。
    """
    tagged = []
    # 标记已知单词
    for i, token in enumerate(tokens):
        tagged.append([token, lexicon.get(token, i == 0 and lexicon.get(token.lower()) or None)])
    # 标记未知单词
    for i, (token, tag) in enumerate(tagged):
        prev, next = (None, None), (None, None)
        if i > 0:
            prev = tagged[i-1]
        if i < len(tagged) - 1:
            next = tagged[i+1]
        if tag is None or token in (model is not None and model.unknown or ()):
            # 使用语言模型 (例如SLP)
            if model is not None:
                tagged[i] = model.apply([token, None], prev, next)
            # 使用NNP标识大写字母
            elif token.istitle() and language != "de":
                tagged[i] = [token, default[1]]
            # 使用CD标识数字
            elif CD.match(token) is not None:
                tagged[i] = [token, default[2]]

            elif morphology is not None:
                tagged[i] = morphology.apply([token, default[0]], prev, next)

            elif language == "en":
                tagged[i] = _suffix_rules([token, default[0]])

            else:
                tagged[i] = [token, default[0]]
    # 使用上下文标识字词
    if context is not None and model is None:
        tagged = context.apply(tagged)
    # 标识命名实体
    if entities is not None:
        tagged = entities.apply(tagged)

    if map is not None:
        tagged = [list(map(token, tag)) or [token, default[0]] for token, tag in tagged]
    return tagged

SEPARATOR = "/"

NN = r"NN|NNS|NNP|NNPS|NNPS?\-[A-Z]{3,4}|PR|PRP|PRP\$"
VB = r"VB|VBD|VBG|VBN|VBP|VBZ"
JJ = r"JJ|JJR|JJS"
RB = r"(?<!W)RB|RBR|RBS"

CHUNKS = [[
    (  "NP", re.compile(r"(("+NN+")/)*((DT|CD|CC|CJ)/)*(("+RB+"|"+JJ+")/)*(("+NN+")/)+")),
    (  "VP", re.compile(r"(((MD|"+RB+")/)*(("+VB+")/)+)+")),
    (  "VP", re.compile(r"((MD)/)")),
    (  "PP", re.compile(r"((IN|PP|TO)/)+")),
    ("ADJP", re.compile(r"((CC|CJ|"+RB+"|"+JJ+")/)*(("+JJ+")/)+")),
    ("ADVP", re.compile(r"(("+RB+"|WRB)/)+")),
], [
    (  "NP", re.compile(r"(("+NN+")/)*((DT|CD|CC|CJ)/)*(("+RB+"|"+JJ+")/)*(("+NN+")/)+(("+RB+"|"+JJ+")/)*")),
    (  "VP", re.compile(r"(((MD|"+RB+")/)*(("+VB+")/)+(("+RB+")/)*)+")),
    (  "VP", re.compile(r"((MD)/)")),
    (  "PP", re.compile(r"((IN|PP|TO)/)+")),
    ("ADJP", re.compile(r"((CC|CJ|"+RB+"|"+JJ+")/)*(("+JJ+")/)+")),
    ("ADVP", re.compile(r"(("+RB+"|WRB)/)+")),
]]

CHUNKS[0].insert(1, CHUNKS[0].pop(3))
CHUNKS[1].insert(1, CHUNKS[1].pop(3))

def find_chunks(tagged, language="en"):
    """ 输入为[token, tag]的表单
        输出为[token, tag, chunk]的表单
    """
    chunked = [x for x in tagged]
    tags = "".join("%s%s" % (tag, SEPARATOR) for token, tag in tagged)
    # 根据给定的语言使用日耳曼语或罗马语分块规则
    for tag, rule in CHUNKS[int(language in ("ca", "es", "pt", "fr", "it", "pt", "ro"))]:
        for m in rule.finditer(tags):
            # 在标签字符串中找到块的开头
            i = m.start()
            j = tags[:i].count(SEPARATOR)
            n = m.group(0).count(SEPARATOR)
            for k in range(j, j+n):
                if len(chunked[k]) == 3:
                    continue
                if len(chunked[k]) < 3:
                    # 连词不能是块的开头
                    if k == j and chunked[k][1] in ("CC", "CJ", "KON", "Conj(neven)"):
                        j += 1
                    # 用B-标记块中的第一个token
                    elif k == j:
                        chunked[k].append("B-"+tag)
                    # 标记块中的其余tokens为I-
                    else:
                        chunked[k].append("I-"+tag)
    # 用O-.标记chinks（块之外的token）
    for chink in filter(lambda x: len(x) < 3, chunked):
        chink.append("O")

    for i, (word, tag, chunk) in enumerate(chunked):
        if tag.startswith("RB") and chunk == "B-NP":

            if i < len(chunked)-1 and not chunked[i+1][1].startswith("JJ"):
                chunked[i+0][2] = "B-ADVP"
                chunked[i+1][2] = "B-NP"
    return chunked

def find_prepositions(chunked):
    """ 输入为[token, tag，chunk]的表单
        输出为[token, tag, chunk, preposition]的表单
    """
    # 不属于介词的标记会被标记上O标签。
    for ch in chunked:
        ch.append("O")
    for i, chunk in enumerate(chunked):
        if chunk[2].endswith("PP") and chunk[-1] == "O":

            if i < len(chunked)-1 and \
             (chunked[i+1][2].endswith(("NP", "PP")) or \
              chunked[i+1][1] in ("VBG", "VBN")):
                chunk[-1] = "B-PNP"
                pp = True
                for ch in chunked[i+1:]:
                    if not (ch[2].endswith(("NP", "PP")) or ch[1] in ("VBG", "VBN")):
                        break
                    if ch[2].endswith("PP") and pp:
                        ch[-1] = "I-PNP"
                    if not ch[2].endswith("PP"):
                        ch[-1] = "I-PNP"
                        pp = False
    return chunked

PTB = PENN = "penn"

class Parser:

    def __init__(self, lexicon={}, default=("NN", "NNP", "CD"), language=None):
        """ 使用基于Brill的词性标记器的简单浅层解析器。
            给定的lexicon是已知单词及其词性标签的字典。
            给定的默认标签用于未知单词。
            以大写字母开头的未知单词将标记为NNP。
            仅包含数字和标点符号的未知单词将标记为CD。
        """
        self.lexicon  = lexicon
        self.default  = default
        self.language = language

    def find_tokens(self, string, **kwargs):
        #从给定字符串中返回句子表单。
        #标点符号与每个单词之间用空格分隔。
        
        return find_tokens(text_type(string),
                punctuation = kwargs.get(  "punctuation", PUNCTUATION),
              abbreviations = kwargs.get("abbreviations", ABBREVIATIONS),
                    replace = kwargs.get(      "replace", replacements),
                  linebreak = r"\n{2,}")

    def find_tags(self, tokens, **kwargs):
        # 使用词性标记标识给定的标记表单。
        # 返回tokens表单, 其中每一个token都标识一个[word, tag]表单。
    
        return find_tags(tokens,
                   language = kwargs.get("language", self.language),
                    lexicon = kwargs.get( "lexicon", self.lexicon),
                    default = kwargs.get( "default", self.default),
                        map = kwargs.get(     "map", None))

    def find_chunks(self, tokens, **kwargs):
        #使用块标记标识给定的token表单。
        #有一些标签可以被添加, 例如chunk + preposition标签。
       
        return find_prepositions(
               find_chunks(tokens,
                   language = kwargs.get("language", self.language)))

    def find_prepositions(self, tokens, **kwargs):
        #使用介词名词短语标签标识给定的token表单。
        return find_prepositions(tokens) # See also Parser.find_chunks().

    def find_labels(self, tokens, **kwargs):
        # 使用谓词/谓词标记标识给定的token表单。
        return find_relations(tokens)

    def find_lemmata(self, tokens, **kwargs):
        return [token + [token[0].lower()] for token in tokens]

    def parse(self, s, tokenize=True, tags=True, chunks=True, relations=False, lemmata=False, encoding="utf-8", **kwargs):
        """ 获取一个字符串（句子）并返回一个标记的Unicode字符串。
            输出中的句子用换行符分隔。
            当tokenize=True时，标点符号从单词中分离出来，句子用\n分隔。
            当tags=True时, 解析词性标签(例如NN, VB, IN, ...)。
            当chunks=True时，解析短语块标签（例如NP, VP, PP, PNP, ...)。
            当relations=True时，解析语义角色标签(例如SBJ, OBJ)。
            可选参数将传递给分词器、标记器、分块器、标记器和词形还原器。
        """
        if tokenize:
            s = self.find_tokens(s, **kwargs)
        if isinstance(s, (list, tuple)):
            s = [isinstance(s, basestring) and s.split(" ") or s for s in s]
        if isinstance(s, basestring):
            s = [s.split(" ") for s in s.split("\n")]

        for i in range(len(s)):
            for j in range(len(s[i])):
                if isinstance(s[i][j], binary_type):
                    s[i][j] = decode_string(s[i][j], encoding)

            if tags or chunks or relations or lemmata:
                s[i] = self.find_tags(s[i], **kwargs)
            else:
                s[i] = [[w] for w in s[i]]

            if chunks or relations:
                s[i] = self.find_chunks(s[i], **kwargs)

            if relations:
                s[i] = self.find_labels(s[i], **kwargs)

            if lemmata:
                s[i] = self.find_lemmata(s[i], **kwargs)

        if not kwargs.get("collapse", True) \
            or kwargs.get("split", False):
            return s
        # 构造TaggedString.format.
        format = ["word"]
        if tags:
            format.append("part-of-speech")
        if chunks:
            format.extend(("chunk", "preposition"))
        if relations:
            format.append("relation")
        if lemmata:
            format.append("lemma")

        for i in range(len(s)):
            for j in range(len(s[i])):
                s[i][j][0] = s[i][j][0].replace("/", "&slash;")
                s[i][j] = "/".join(s[i][j])
            s[i] = " ".join(s[i])
        s = "\n".join(s)
        s = TaggedString(unicode(s), format, language=kwargs.get("language", self.language))
        return s

# Pattern.parse()返回一个TaggedString：一个带有“tags”和“language”属性的Unicode字符串。
# pattern.text.tree.Text类使用此属性来确定标记格式，并将标记的字符串转换为嵌套句子、块和Word对象的分析树。

TOKENS = "tokens"

class TaggedString(unicode):

    def __new__(self, string, tags=["word"], language=None):
        if isinstance(string, unicode) and hasattr(string, "tags"):
            tags, language = string.tags, string.language

        if isinstance(string, list):
            string = [[[x.replace("/", "&slash;") for x in token] for token in s] for s in string]
            string = "\n".join(" ".join("/".join(token) for token in s) for s in string)
        s = unicode.__new__(self, string)
        s.tags = list(tags)
        s.language = language
        return s

    def split(self, sep=TOKENS):
        # 返回句子表单，其中每个句子都是token表单，而每个token是单词 + 标签的表单。

        if sep != TOKENS:
            return unicode.split(self, sep)
        if len(self) == 0:
            return []
        return [[[x.replace("&slash;", "/") for x in token.split("/")]
            for token in sentence.split(" ")]
                for sentence in unicode.split(self, "\n")]


class Spelling(lazydict):

    ALPHA = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self, path=""):
        self._path = path

    def load(self):
        for x in _read(self._path):
            x = x.split()
            dict.__setitem__(self, x[0], int(x[1]))

    @property
    def path(self):
        return self._path

    @property
    def language(self):
        return self._language

    @classmethod
    def train(self, s, path="spelling.txt"):
        #计算给定字符串中的单词，并将可能性保存到给定路径。这可用于为Spelling()构造函数生成新模型。
        model = {}
        for w in re.findall("[a-z]+", s.lower()):
            model[w] = w in model and model[w] + 1 or 1
        model = ("%s %s" % (k, v) for k, v in sorted(model.items()))
        model = "\n".join(model)
        f = open(path, "w")
        f.write(model)
        f.close()

    def _edit1(self, w):
       # 返回一组与给定单词的编辑距离为1的单词。
        split = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        delete, transpose, replace, insert = (
            [a + b[1:] for a, b in split if b],
            [a + b[1] + b[0] + b[2:] for a, b in split if len(b) > 1],
            [a + c + b[1:] for a, b in split for c in Spelling.ALPHA if b],
            [a + c + b[0:] for a, b in split for c in Spelling.ALPHA]
        )
        return set(delete + transpose + replace + insert)

    def _edit2(self, w):
    # 返回一组与给定单词的编辑距离为2的单词。
        return set(e2 for e1 in self._edit1(w) for e2 in self._edit1(e1) if e2 in self)

    def _known(self, words=[]):
        # 返回按已知单词筛选的给定单词列表。
        return set(w for w in words if w in self)

    def suggest(self, w):
        #根据与给定单词编辑距离为1-2的已知单词的概率，返回给定单词的(word, confidence)拼写更正列表。
        if len(self) == 0:
            self.load()
        if len(w) == 1:
            return [(w, 1.0)] 
        if w in PUNCTUATION:
            return [(w, 1.0)] 
        if w in string.whitespace:
            return [(w, 1.0)] 
        if w.replace(".", "").isdigit():
            return [(w, 1.0)] 
        candidates = self._known([w]) \
                  or self._known(self._edit1(w)) \
                  or self._known(self._edit2(w)) \
                  or [w]
        candidates = [(self.get(c, 0.0), c) for c in candidates]
        s = float(sum(p for p, word in candidates) or 1)
        candidates = sorted(((p / s, word) for p, word in candidates), reverse=True)
        if w.istitle(): 
            candidates = [(word.title(), p) for p, word in candidates]
        else:
            candidates = [(word, p) for p, word in candidates]
        return candidates
