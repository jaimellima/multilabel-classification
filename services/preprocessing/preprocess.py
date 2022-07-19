
import re
from random import sample

import pandas as pd
import numpy as np


class Preprocess:
    def __init__(self):
        self._dataset = None
        self._X = None
        self._y = None
        self._stopwords = None

    def set_dataset(self, origin_csv_file, n_sample):
        self._dataset = pd.read_csv(origin_csv_file)
        self._dataset = self._dataset.sample(n=n_sample)

    def get_dataset(self):
        return self._dataset

    def set_X(self, x_columns):
        self._X = self._dataset[x_columns].apply(lambda row: ''.join(row.values.astype(str)), axis=1)

    def get_X(self):
        return self._X

    def get_y(self):
        return self._y

    def set_y(self, y_columns):
        self._y = self._dataset[y_columns]

    def remove_punctuations(self, text):
        cleaned_text = text.lower()
        cleaned_text = re.findall("[a-zA-Z0-9]+", cleaned_text)
        return cleaned_text

    def remove_stopwords(self, text):
        cleaned_text = [word for word in text if not word in self._stopwords]
        return cleaned_text

    def get_preprocessed_text(self):
        documents = np.array(self._X)
        X_preprocessed = []
        for text in documents:
            cleaned_text = self.remove_punctuations(text)
            cleaned_text = self.remove_stopwords(cleaned_text)
            cleaned_text = " ".join([word for word in cleaned_text])
            #lemma_text = self.lemmatize_tokenize(self.__nlp_model, cleaned_text)
            #final_text = " ".join([word for word in lemma_text])
            X_preprocessed.append(cleaned_text)
        return X_preprocessed

    def vectorization(self):
        #strategy??? abertas para extensão, mas fechadas para modificação
        pass
        
    


        


