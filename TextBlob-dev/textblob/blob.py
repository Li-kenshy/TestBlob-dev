# -*- coding: utf-8 -*-
from __future__ import unicode_literals, absolute_import
import sys
import json
import warnings
from collections import defaultdict

import nltk

from textblob.decorators import cached_property, requires_nltk_corpus
from textblob.utils import lowerstrip, PUNCTUATION_REGEX
from textblob.inflect import singularize as _singularize, pluralize as _pluralize
from textblob.mixins import BlobComparableMixin, StringlikeMixin
from textblob.compat import unicode, basestring
from textblob.base import (BaseNPExtractor, BaseTagger, BaseTokenizer,
                       BaseSentimentAnalyzer, BaseParser)
from textblob.np_extractors import FastNPExtractor
from textblob.taggers import NLTKTagger
from textblob.tokenizers import WordTokenizer, sent_tokenize, word_tokenize
from textblob.sentiments import PatternAnalyzer
from textblob.parsers import PatternParser
from textblob.translate import Translator
from textblob.en import suggest

# Wordnet interface
# NOTE: textblob.wordnet is not imported so that the wordnet corpus can be lazy-loaded
_wordnet = nltk.corpus.wordnet

def _penn_to_wordnet(tag):
    if tag in ("NN", "NNS", "NNP", "NNPS"):
        return _wordnet.NOUN
    if tag in ("JJ", "JJR", "JJS"):
        return _wordnet.ADJ
    if tag in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
        return _wordnet.VERB
    if tag in ("RB", "RBR", "RBS"):
        return _wordnet.ADV
    return None

class Word(unicode):

    translator = Translator()

    def __new__(cls, string, pos_tag=None):
        return super(Word, cls).__new__(cls, string)

    def __init__(self, string, pos_tag=None):
        self.string = string
        self.pos_tag = pos_tag

    def __repr__(self):
        return repr(self.string)

    def __str__(self):
        return self.string

    def singularize(self):
        return Word(_singularize(self.string))

    def pluralize(self):
        return Word(_pluralize(self.string))

    def translate(self, from_lang='auto', to="en"):
        warnings.warn(
            'Word.translate is deprecated and will be removed in a future release. '
            'Use the official Google Translate API instead.',
            DeprecationWarning
        )
        return self.translator.translate(self.string,
                                         from_lang=from_lang, to_lang=to)

    def detect_language(self):
        warnings.warn(
            'Word.detect_language is deprecated and will be removed in a future release. '
            'Use the official Google Translate API instead.',
            DeprecationWarning
        )
        return self.translator.detect(self.string)

    def spellcheck(self):
        return suggest(self.string)

    def correct(self):
        return Word(self.spellcheck()[0][0])

    @cached_property
    @requires_nltk_corpus
    def lemma(self):
        return self.lemmatize(pos=self.pos_tag)

    @requires_nltk_corpus
    def lemmatize(self, pos=None):
        if pos is None:
            tag = _wordnet.NOUN
        elif pos in _wordnet._FILEMAP.keys():
            tag = pos
        else:
            tag = _penn_to_wordnet(pos)
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return lemmatizer.lemmatize(self.string, tag)

    PorterStemmer = nltk.stem.porter.PorterStemmer()
    LancasterStemmer = nltk.stem.lancaster.LancasterStemmer()
    SnowballStemmer = nltk.stem.snowball.SnowballStemmer("english")

    def stem(self, stemmer=PorterStemmer):
        return stemmer.stem(self.string)

    @cached_property
    def synsets(self):
        return self.get_synsets(pos=None)

    @cached_property
    def definitions(self):
        return self.define(pos=None)

    def get_synsets(self, pos=None):
        return _wordnet.synsets(self.string, pos)

    def define(self, pos=None):
        return [syn.definition() for syn in self.get_synsets(pos=pos)]


class WordList(list):


    def __init__(self, collection):
        super(WordList, self).__init__([Word(w) for w in collection])

    def __str__(self):
        return super(WordList, self).__repr__()

    def __repr__(self):
        class_name = self.__class__.__name__
        return '{cls}({lst})'.format(cls=class_name, lst=super(WordList, self).__repr__())

    def __getitem__(self, key):
        item = super(WordList, self).__getitem__(key)
        if isinstance(key, slice):
            return self.__class__(item)
        else:
            return item

    def __getslice__(self, i, j):
        return self.__class__(super(WordList, self).__getslice__(i, j))

    def __setitem__(self, index, obj):
        if isinstance(obj, basestring):
            super(WordList, self).__setitem__(index, Word(obj))
        else:
            super(WordList, self).__setitem__(index, obj)

    def count(self, strg, case_sensitive=False, *args, **kwargs):
        if not case_sensitive:
            return [word.lower() for word in self].count(strg.lower(), *args,
                    **kwargs)
        return super(WordList, self).count(strg, *args, **kwargs)

    def append(self, obj):
        if isinstance(obj, basestring):
            super(WordList, self).append(Word(obj))
        else:
            super(WordList, self).append(obj)

    def extend(self, iterable):
        for e in iterable:
            self.append(e)

    def upper(self):
        return self.__class__([word.upper() for word in self])

    def lower(self):
        return self.__class__([word.lower() for word in self])

    def singularize(self):
        return self.__class__([word.singularize() for word in self])

    def pluralize(self):
        return self.__class__([word.pluralize() for word in self])

    def lemmatize(self):
        return self.__class__([word.lemmatize() for word in self])

    def stem(self, *args, **kwargs):
        return self.__class__([word.stem(*args, **kwargs) for word in self])


def _validated_param(obj, name, base_class, default, base_class_name=None):
    base_class_name = base_class_name if base_class_name else base_class.__name__
    if obj is not None and not isinstance(obj, base_class):
        raise ValueError('{name} must be an instance of {cls}'
                         .format(name=name, cls=base_class_name))
    return obj or default


def _initialize_models(obj, tokenizer, pos_tagger,
                       np_extractor, analyzer, parser, classifier):
    obj.tokenizer = _validated_param(tokenizer, "tokenizer",
                                    base_class=(BaseTokenizer, nltk.tokenize.api.TokenizerI),
                                    default=BaseBlob.tokenizer,
                                    base_class_name="BaseTokenizer")
    obj.np_extractor = _validated_param(np_extractor, "np_extractor",
                                        base_class=BaseNPExtractor,
                                        default=BaseBlob.np_extractor)
    obj.pos_tagger = _validated_param(pos_tagger, "pos_tagger",
                                        BaseTagger, BaseBlob.pos_tagger)
    obj.analyzer = _validated_param(analyzer, "analyzer",
                                     BaseSentimentAnalyzer, BaseBlob.analyzer)
    obj.parser = _validated_param(parser, "parser", BaseParser, BaseBlob.parser)
    obj.classifier = classifier


class BaseBlob(StringlikeMixin, BlobComparableMixin):
    np_extractor = FastNPExtractor()
    pos_tagger = NLTKTagger()
    tokenizer = WordTokenizer()
    translator = Translator()
    analyzer = PatternAnalyzer()
    parser = PatternParser()

    def __init__(self, text, tokenizer=None,
                pos_tagger=None, np_extractor=None, analyzer=None,
                parser=None, classifier=None, clean_html=False):
        if not isinstance(text, basestring):
            raise TypeError('The `text` argument passed to `__init__(text)` '
                            'must be a string, not {0}'.format(type(text)))
        if clean_html:
            raise NotImplementedError("clean_html has been deprecated. "
                                    "To remove HTML markup, use BeautifulSoup's "
                                    "get_text() function")
        self.raw = self.string = text
        self.stripped = lowerstrip(self.raw, all=True)
        _initialize_models(self, tokenizer, pos_tagger, np_extractor, analyzer,
                           parser, classifier)

    @cached_property
    def words(self):
        return WordList(word_tokenize(self.raw, include_punc=False))

    @cached_property
    def tokens(self):
        return WordList(self.tokenizer.tokenize(self.raw))

    def tokenize(self, tokenizer=None):
        t = tokenizer if tokenizer is not None else self.tokenizer
        return WordList(t.tokenize(self.raw))

    def parse(self, parser=None):
        p = parser if parser is not None else self.parser
        return p.parse(self.raw)

    def classify(self):
        if self.classifier is None:
            raise NameError("This blob has no classifier. Train one first!")
        return self.classifier.classify(self.raw)

    @cached_property
    def sentiment(self):
        return self.analyzer.analyze(self.raw)

    @cached_property
    def sentiment_assessments(self):
        return self.analyzer.analyze(self.raw, keep_assessments=True)

    @cached_property
    def polarity(self):
        return PatternAnalyzer().analyze(self.raw)[0]

    @cached_property
    def subjectivity(self):
        return PatternAnalyzer().analyze(self.raw)[1]

    @cached_property
    def noun_phrases(self):
        return WordList([phrase.strip().lower()
                        for phrase in self.np_extractor.extract(self.raw)
                        if len(phrase) > 1])

    @cached_property
    def pos_tags(self):
        if isinstance(self, TextBlob):
            return [val for sublist in [s.pos_tags for s in self.sentences] for val in sublist]
        else:
            return [(Word(unicode(word), pos_tag=t), unicode(t))
                    for word, t in self.pos_tagger.tag(self)
                    if not PUNCTUATION_REGEX.match(unicode(t))]

    tags = pos_tags

    @cached_property
    def word_counts(self):
        counts = defaultdict(int)
        stripped_words = [lowerstrip(word) for word in self.words]
        for word in stripped_words:
            counts[word] += 1
        return counts

    @cached_property
    def np_counts(self):
        counts = defaultdict(int)
        for phrase in self.noun_phrases:
            counts[phrase] += 1
        return counts

    def ngrams(self, n=3):
        if n <= 0:
            return []
        grams = [WordList(self.words[i:i + n])
                            for i in range(len(self.words) - n + 1)]
        return grams

    def translate(self, from_lang="auto", to="en"):
        warnings.warn(
            'TextBlob.translate is deprecated and will be removed in a future release. '
            'Use the official Google Translate API instead.',
            DeprecationWarning
        )
        return self.__class__(self.translator.translate(self.raw,
                              from_lang=from_lang, to_lang=to))

    def detect_language(self):
        warnings.warn(
            'TextBlob.detext_translate is deprecated and will be removed in a future release. '
            'Use the official Google Translate API instead.',
            DeprecationWarning
        )
        return self.translator.detect(self.raw)

    def correct(self):
        tokens = nltk.tokenize.regexp_tokenize(self.raw, r"\w+|[^\w\s]|\s")
        corrected = (Word(w).correct() for w in tokens)
        ret = ''.join(corrected)
        return self.__class__(ret)

    def _cmpkey(self):
        return self.raw

    def _strkey(self):
        return self.raw

    def __hash__(self):
        return hash(self._cmpkey())

    def __add__(self, other):
        if isinstance(other, basestring):
            return self.__class__(self.raw + other)
        elif isinstance(other, BaseBlob):
            return self.__class__(self.raw + other.raw)
        else:
            raise TypeError('Operands must be either strings or {0} objects'
                .format(self.__class__.__name__))

    def split(self, sep=None, maxsplit=sys.maxsize):
        return WordList(self._strkey().split(sep, maxsplit))


class TextBlob(BaseBlob):

    @cached_property
    def sentences(self):
        return self._create_sentence_objects()

    @cached_property
    def words(self):
        return WordList(word_tokenize(self.raw, include_punc=False))

    @property
    def raw_sentences(self):
        return [sentence.raw for sentence in self.sentences]

    @property
    def serialized(self):
        return [sentence.dict for sentence in self.sentences]

    def to_json(self, *args, **kwargs):
        return json.dumps(self.serialized, *args, **kwargs)

    @property
    def json(self):
        return self.to_json()

    def _create_sentence_objects(self):
        sentence_objects = []
        sentences = sent_tokenize(self.raw)
        char_index = 0  
        for sent in sentences:
            start_index = self.raw.index(sent, char_index)
            char_index += len(sent)
            end_index = start_index + len(sent)
            s = Sentence(sent, start_index=start_index, end_index=end_index,
                tokenizer=self.tokenizer, np_extractor=self.np_extractor,
                pos_tagger=self.pos_tagger, analyzer=self.analyzer,
                parser=self.parser, classifier=self.classifier)
            sentence_objects.append(s)
        return sentence_objects


class Sentence(BaseBlob):

    def __init__(self, sentence, start_index=0, end_index=None, *args, **kwargs):
        super(Sentence, self).__init__(sentence, *args, **kwargs)
        self.start = self.start_index = start_index
        self.end = self.end_index = end_index or len(sentence) - 1

    @property
    def dict(self):
        return {
            'raw': self.raw,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'stripped': self.stripped,
            'noun_phrases': self.noun_phrases,
            'polarity': self.polarity,
            'subjectivity': self.subjectivity,
        }


class Blobber(object):


    np_extractor = FastNPExtractor()
    pos_tagger = NLTKTagger()
    tokenizer = WordTokenizer()
    analyzer = PatternAnalyzer()
    parser = PatternParser()

    def __init__(self, tokenizer=None, pos_tagger=None, np_extractor=None,
                analyzer=None, parser=None, classifier=None):
        _initialize_models(self, tokenizer, pos_tagger, np_extractor, analyzer,
                            parser, classifier)

    def __call__(self, text):
        return TextBlob(text, tokenizer=self.tokenizer, pos_tagger=self.pos_tagger,
                        np_extractor=self.np_extractor, analyzer=self.analyzer,
                        parser=self.parser,
                        classifier=self.classifier)

    def __repr__(self):
        classifier_name = self.classifier.__class__.__name__ + "()" if self.classifier else "None"
        return ("Blobber(tokenizer={0}(), pos_tagger={1}(), "
                    "np_extractor={2}(), analyzer={3}(), parser={4}(), classifier={5})")\
                    .format(self.tokenizer.__class__.__name__,
                            self.pos_tagger.__class__.__name__,
                            self.np_extractor.__class__.__name__,
                            self.analyzer.__class__.__name__,
                            self.parser.__class__.__name__,
                            classifier_name)

    __str__ = __repr__
