# -*- coding: utf-8 -*-

MISSING_CORPUS_MESSAGE = """
Looks like you are missing some required data for this feature.

To download the necessary data, simply run

    python -m textblob.download_corpora

or use the NLTK downloader to download the missing data: http://nltk.org/data.html
If this doesn't fix the problem, file an issue at https://github.com/sloria/TextBlob/issues.
"""

class TextBlobError(Exception):
    pass


TextBlobException = TextBlobError 

class MissingCorpusError(TextBlobError):

    def __init__(self, message=MISSING_CORPUS_MESSAGE, *args, **kwargs):
        super(MissingCorpusError, self).__init__(message, *args, **kwargs)


MissingCorpusException = MissingCorpusError 

class DeprecationError(TextBlobError):
    pass

class TranslatorError(TextBlobError):
    pass

class NotTranslated(TranslatorError):
    pass

class FormatError(TextBlobError):
    pass
