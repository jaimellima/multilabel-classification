
from __future__ import annotations


import sys
sys.path.append('./')

import re
from random import sample
import pandas as pd
import numpy as np
from abc import ABC, abstractclassmethod
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from vectorize_strategy import TfVectorStrategy, TfIdfVectorStrategy


class Preprocessing:
    def __init__(self, stopwords, vectorize_strategy: VectorizaStrategy, nlp_model=None, ngram_range=(1,3)):
        self._stopwords = stopwords
        self._vectorize_strategy = vectorize_strategy
        self._nlp_model = nlp_model
        self._ngram_range = ngram_range

    def remove_punctuations(self, text):
        text_lower = [word.lower() for word in text]
        cleaned_text = re.findall("[a-zA-Z0-9]+", str(text_lower))
        return cleaned_text

    def remove_stopwords(self, text):
        cleaned_text = [word for word in text if not word in self._stopwords]
        cleaned_text = [" ".join([word for word in cleaned_text])]
        return cleaned_text
            
    def lemmatize_tokenize(self, cleaned_text): pass
            #final_text = " ".join([word for word in lemma_text])
            #X_preprocessed.append(cleaned_text)
        #return X_preprocessed

    def vectorization(self, X_preprocessed):
        return self._vectorize_strategy.vectorize(X_preprocessed=X_preprocessed)
   
#teste
if __name__=="__main__":
    stopwords = None
    X_preprocessed = [["isso", "é", "um", "teste"], ["isso", "é", "outro", "teste2"]]
    #vectorize_strategy = TfVectorStrategy()
    vectorize_strategy = TfIdfVectorStrategy()
    preprocess = Preprocessing(stopwords=stopwords, vectorize_strategy=vectorize_strategy)
    X_vectorized = preprocess.vectorization(np.array(X_preprocessed, dtype=str))
    print(X_vectorized)










        


