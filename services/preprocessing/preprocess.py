
import re
from random import sample

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from __future__ import annotations


class Preprocess:
    def __init__(self, vectorization_strategy: VectorizationStrategy):
        self._dataset = None
        self._stopwords = None
        self._vectorization_strategy = vectorization_strategy


    def remove_punctuations(self, text):
        cleaned_text = text.lower()
        cleaned_text = re.findall("[a-zA-Z0-9]+", cleaned_text)
        return cleaned_text

    def remove_stopwords(self, text):
        X_preprocessed = []
        for text in self._dataset.get_documents():
            cleaned_text = self.remove_punctuations(text)
            cleaned_text = self.remove_stopwords(cleaned_text)
            cleaned_text = " ".join([word for word in cleaned_text])
            #lemma_text = self.lemmatize_tokenize(self.__nlp_model, cleaned_text)
            #final_text = " ".join([word for word in lemma_text])
            X_preprocessed.append(cleaned_text)
        return X_preprocessed

    def vectorization(self, text, vectorization_strategy):
        return self._vectorization_strategy.vectorize(text)
        
        
    
class VectorizationStrategy():
    @abstractmethod
    def vectorize(self, text): pass


        


