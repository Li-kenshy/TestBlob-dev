#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import nltk

MIN_CORPORA = [
    'brown',  
    'punkt',  
    'wordnet',  
    'averaged_perceptron_tagger',  
]

ADDITIONAL_CORPORA = [
    'conll2000',  
    'movie_reviews', 
]

ALL_CORPORA = MIN_CORPORA + ADDITIONAL_CORPORA

def download_lite():
    for each in MIN_CORPORA:
        nltk.download(each)


def download_all():
    for each in ALL_CORPORA:
        nltk.download(each)


def main():
    if 'lite' in sys.argv:
        download_lite()
    else:
        download_all()
    print("Finished.")


if __name__ == '__main__':
    main()
