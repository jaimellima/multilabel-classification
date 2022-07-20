
import re
from random import sample
from __future__ import annotations
import pandas as pd
import numpy as np
from abc import ABC, abstractclassmethod


class Preprocess:
    def __init__(self, stopwords, vectorize_strategy: VectorizaStrategy, nlp_model=None):
        self._stopwords = stopwords
        self._vectorize_strategy = vectorize_strategy
        self._nlp_model = nlp_model

    def remove_punctuations(self, text):
        cleaned_text = text.lower()
        cleaned_text = re.findall("[a-zA-Z0-9]+", cleaned_text)
        return cleaned_text

    def remove_stopwords(self, text):
        cleaned_text = [word for word in text if not word in self._stopwords]
        cleaned_text = " ".join([word for word in cleaned_text])
        return cleaned_text
            
    def lemmatize_tokenize(self, cleaned_text): pass
            #final_text = " ".join([word for word in lemma_text])
            #X_preprocessed.append(cleaned_text)
        #return X_preprocessed

    def vectorization(self, text):
        return self._vectorize_strategy.vectorize(text)
        
        
    
class VectorizeStrategy(ABC):
    @abstractclassmethod
    def vectorize(self, text):
        pass

        


