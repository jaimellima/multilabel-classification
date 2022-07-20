from __future__ import annotations

import sys
sys.path.append('./')

from abc import ABC, abstractclassmethod
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class VectorStrategy(ABC):
    @abstractclassmethod
    def vectorize(self, X_preprocessed): pass
    #def vectorize(self, X_preprocessed): pass

class TfVectorStrategy(VectorStrategy):
    def __init__(self, ngram_range=(1,3)):
        self._ngram_range=ngram_range

    def vectorize(self, X_preprocessed):
        print("Using Term Frequency strategy...n-gram range: {}".format(self._ngram_range))
        X_vectorized = None
        vectorizer = CountVectorizer(analyzer='word', ngram_range=self._ngram_range)
        X_vectorized = vectorizer.fit_transform(X_preprocessed)
        return X_vectorized

class TfIdfVectorStrategy(VectorStrategy):
    def __init__(self, ngram_range=(1,3)):
        self._ngram_range=ngram_range

    def vectorize(self, X_preprocessed, ngram_range=None):
        print("Using TF-IDF strategy...")
        vetorizer = TfidfVectorizer()
        X_vectorized = vetorizer.fit_transform(X_preprocessed)
        return X_vectorized

class Doc2Vec(VectorStrategy):
    pass