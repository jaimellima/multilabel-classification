
import sys
sys.path.append('./')

from services.preprocessing.preprocessing import Preprocessing
from preprocessing import Preprocessing
from vectorize_strategy import TfIdfVectorStrategy, TfVectorStrategy
import numpy as np


class Preprocessor:
    def __init__(self, stopwords, vectorize_strategy):
        self._stopwords = stopwords
        self._vectorize_strategy = vectorize_strategy
        self._preprocessing = self.get_preprocessing()
        
    def get_preprocessing(self):
        preprocessing = Preprocessing(stopwords=self._stopwords, vectorize_strategy=self._vectorize_strategy)
        return preprocessing

    def execute(self, documents):
        #executa os passos do preprocessamento do texto.
        cleaned_texts = []
        for document in documents:
            text = self._preprocessing.remove_punctuations(document)
            text = self._preprocessing.remove_stopwords(text)
            cleaned_texts.append(np.array(text))
        matrix = self._preprocessing.vectorization(text)
        return matrix
    
#TESTING
if __name__=="__main__":
    from spacy.lang.en.stop_words import STOP_WORDS
    vectorize_strategy = TfVectorStrategy(ngram_range=(1,3))
    preprocessor = Preprocessor(stopwords=STOP_WORDS, vectorize_strategy=vectorize_strategy)
    X_preprocessed = [["the", "boy", "see", "the", "car"], ["the", "girl", "plays", "football"]]
    matrix = preprocessor.execute(X_preprocessed)
    print(matrix)

    