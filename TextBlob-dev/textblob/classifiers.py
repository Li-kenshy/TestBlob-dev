# -*- coding: utf-8 -*-
from __future__ import absolute_import
from itertools import chain

import nltk

from textblob.compat import basestring
from textblob.decorators import cached_property
from textblob.exceptions import FormatError
from textblob.tokenizers import word_tokenize
from textblob.utils import strip_punc, is_filelike
import textblob.formats as formats

### Basic feature extractors ###


def _get_words_from_dataset(dataset):

    def tokenize(words):
        if isinstance(words, basestring):
            return word_tokenize(words, include_punc=False)
        else:
            return words
    all_words = chain.from_iterable(tokenize(words) for words, _ in dataset)
    return set(all_words)

def _get_document_tokens(document):
    if isinstance(document, basestring):
        tokens = set((strip_punc(w, all=False)
                    for w in word_tokenize(document, include_punc=False)))
    else:
        tokens = set(strip_punc(w, all=False) for w in document)
    return tokens

def basic_extractor(document, train_set):

    try:
        el_zero = next(iter(train_set))  
    except StopIteration:
        return {}
    if isinstance(el_zero, basestring):
        word_features = [w for w in chain([el_zero], train_set)]
    else:
        try:
            assert(isinstance(el_zero[0], basestring))
            word_features = _get_words_from_dataset(chain([el_zero], train_set))
        except Exception:
            raise ValueError('train_set is probably malformed.')

    tokens = _get_document_tokens(document)
    features = dict(((u'contains({0})'.format(word), (word in tokens))
                                            for word in word_features))
    return features


def contains_extractor(document):
    tokens = _get_document_tokens(document)
    features = dict((u'contains({0})'.format(w), True) for w in tokens)
    return features

##### CLASSIFIERS #####

class BaseClassifier(object):

    def __init__(self, train_set, feature_extractor=basic_extractor, format=None, **kwargs):
        self.format_kwargs = kwargs
        self.feature_extractor = feature_extractor
        if is_filelike(train_set):
            self.train_set = self._read_data(train_set, format)
        else: 
            self.train_set = train_set
        self._word_set = _get_words_from_dataset(self.train_set)  
        self.train_features = None

    def _read_data(self, dataset, format=None):
        if not format:
            format_class = formats.detect(dataset)
            if not format_class:
                raise FormatError('Could not automatically detect format for the given '
                                  'data source.')
        else:
            registry = formats.get_registry()
            if format not in registry.keys():
                raise ValueError("'{0}' format not supported.".format(format))
            format_class = registry[format]
        return format_class(dataset, **self.format_kwargs).to_iterable()

    @cached_property
    def classifier(self):
        raise NotImplementedError('Must implement the "classifier" property.')

    def classify(self, text):
        raise NotImplementedError('Must implement a "classify" method.')

    def train(self, labeled_featureset):
        raise NotImplementedError('Must implement a "train" method.')

    def labels(self):
        raise NotImplementedError('Must implement a "labels" method.')

    def extract_features(self, text):
        try:
            return self.feature_extractor(text, self._word_set)
        except (TypeError, AttributeError):
            return self.feature_extractor(text)


class NLTKClassifier(BaseClassifier):
    nltk_class = None

    def __init__(self, train_set,
                 feature_extractor=basic_extractor, format=None, **kwargs):
        super(NLTKClassifier, self).__init__(train_set, feature_extractor, format, **kwargs)
        self.train_features = [(self.extract_features(d), c) for d, c in self.train_set]

    def __repr__(self):
        class_name = self.__class__.__name__
        return "<{cls} trained on {n} instances>".format(cls=class_name,
                                                        n=len(self.train_set))

    @cached_property
    def classifier(self):
        try:
            return self.train()
        except AttributeError:  # nltk_class has not been defined
            raise ValueError("NLTKClassifier must have a nltk_class"
                            " variable that is not None.")

    def train(self, *args, **kwargs):
        try:
            self.classifier = self.nltk_class.train(self.train_features,
                                                    *args, **kwargs)
            return self.classifier
        except AttributeError:
            raise ValueError("NLTKClassifier must have a nltk_class"
                            " variable that is not None.")

    def labels(self):
        return self.classifier.labels()

    def classify(self, text):
        text_features = self.extract_features(text)
        return self.classifier.classify(text_features)

    def accuracy(self, test_set, format=None):
        if is_filelike(test_set):
            test_data = self._read_data(test_set, format)
        else:  # test_set is a list of tuples
            test_data = test_set
        test_features = [(self.extract_features(d), c) for d, c in test_data]
        return nltk.classify.accuracy(self.classifier, test_features)

    def update(self, new_data, *args, **kwargs):
        self.train_set += new_data
        self._word_set.update(_get_words_from_dataset(new_data))
        self.train_features = [(self.extract_features(d), c)
                                for d, c in self.train_set]
        try:
            self.classifier = self.nltk_class.train(self.train_features,
                                                    *args, **kwargs)
        except AttributeError:  # Descendant has not defined nltk_class
            raise ValueError("NLTKClassifier must have a nltk_class"
                            " variable that is not None.")
        return True


class NaiveBayesClassifier(NLTKClassifier):

    nltk_class = nltk.classify.NaiveBayesClassifier

    def prob_classify(self, text):
        text_features = self.extract_features(text)
        return self.classifier.prob_classify(text_features)

    def informative_features(self, *args, **kwargs):
        return self.classifier.most_informative_features(*args, **kwargs)

    def show_informative_features(self, *args, **kwargs):
        return self.classifier.show_most_informative_features(*args, **kwargs)


class DecisionTreeClassifier(NLTKClassifier):

    nltk_class = nltk.classify.decisiontree.DecisionTreeClassifier

    def pretty_format(self, *args, **kwargs):
        return self.classifier.pretty_format(*args, **kwargs)

    # Backwards-compat
    pprint = pretty_format

    def pseudocode(self, *args, **kwargs):
        return self.classifier.pseudocode(*args, **kwargs)


class PositiveNaiveBayesClassifier(NLTKClassifier):

    nltk_class = nltk.classify.PositiveNaiveBayesClassifier

    def __init__(self, positive_set, unlabeled_set,
                feature_extractor=contains_extractor,
                positive_prob_prior=0.5, **kwargs):
        self.feature_extractor = feature_extractor
        self.positive_set = positive_set
        self.unlabeled_set = unlabeled_set
        self.positive_features = [self.extract_features(d)
                                    for d in self.positive_set]
        self.unlabeled_features = [self.extract_features(d)
                                    for d in self.unlabeled_set]
        self.positive_prob_prior = positive_prob_prior

    def __repr__(self):
        class_name = self.__class__.__name__
        return "<{cls} trained on {n_pos} labeled and {n_unlabeled} unlabeled instances>"\
                        .format(cls=class_name, n_pos=len(self.positive_set),
                                n_unlabeled=len(self.unlabeled_set))

    # Override
    def train(self, *args, **kwargs):
        self.classifier = self.nltk_class.train(self.positive_features,
                                                self.unlabeled_features,
                                                self.positive_prob_prior)
        return self.classifier

    def update(self, new_positive_data=None,
               new_unlabeled_data=None, positive_prob_prior=0.5,
               *args, **kwargs):
        self.positive_prob_prior = positive_prob_prior
        if new_positive_data:
            self.positive_set += new_positive_data
            self.positive_features += [self.extract_features(d)
                                            for d in new_positive_data]
        if new_unlabeled_data:
            self.unlabeled_set += new_unlabeled_data
            self.unlabeled_features += [self.extract_features(d)
                                            for d in new_unlabeled_data]
        self.classifier = self.nltk_class.train(self.positive_features,
                                                self.unlabeled_features,
                                                self.positive_prob_prior,
                                                *args, **kwargs)
        return True


class MaxEntClassifier(NLTKClassifier):
    __doc__ = nltk.classify.maxent.MaxentClassifier.__doc__
    nltk_class = nltk.classify.maxent.MaxentClassifier

    def prob_classify(self, text):
        feats = self.extract_features(text)
        return self.classifier.prob_classify(feats)
