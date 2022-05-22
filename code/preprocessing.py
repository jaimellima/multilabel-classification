import spacy
import en_core_web_lg
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer

class Preprocessing:
    
    def __init__(self, nlp_model, stop_words = STOP_WORDS):
        self.__stop_words = stop_words
        self.__nlp_model = nlp_model
  
    def remove_punctuations(self, text):
        text = text.lower()
        cleaned_text = re.findall("[a-zA-Z]+", text)
        return cleaned_text

    def remove_stop_words(self, text):
        cleaned_text = [word for word in text if not word in self.__stop_words]
        return cleaned_text

    def lemmatize_tokenize(self, nlp_model: spacy, text: str):
        lemmas = []
        doc = nlp_model(text)
        for token in doc:
            if not token.is_stop:
                lemmas.append(token.lemma_)
        return lemmas

    def tf_idf_vectorization(self, documents):
        #recebe um vetor/conjunto de documento e retorna uma matriz TF-IDF
        print("Getting TF-IDF...")
        print("Resetting indexes...")
        vetorizer = TfidfVectorizer()
        X_tfidf = vetorizer.fit_transform(documents)
        X_tfidf_dense = X_tfidf.todense()
        pd.DataFrame(X_tfidf_dense).to_csv("tf_idf.csv")
        return X_tfidf_dense

    def doc2vec_spacy(self, documents, nlp_model):
        #recebe um vetor/conjunto de documento e retorna uma matriz com os vetores correspondentes a cada documento.
        vectors = []
        for text in documents:
            doc = nlp_model(text)
            vectors.append(doc.vector)
        return vectors
        
    #Termômetro para a Wisard com base nos valores de mínimo e máximo
    def thermometerEncoderMinMax(self,X, size, min=0, max=1):
        X = np.asarray(X)
        if X.ndim == 0:
            f = lambda i: X >= min + i*(max - min)/size
        elif X.ndim == 1:
            f = lambda i, j: X[j] >= min + i*(max - min)/size
        else:
            f = lambda i, j, k: X[k, j] >= min + i*(max - min)/size 
        return  np.fromfunction(f, (size, *reversed(X.shape)), dtype=int).astype(int)

    #Termômetro para a Wisard com base nos valores de média do vetor
    def thermometerEncoder(self, vector, mean):
        vector = np.asarray(vector)
        vector[vector<mean] = 0
        vector[vector>=mean] = 1
        return vector

    #cria o flatten a partir do dados binarizados pelo termômetro.
    def flatten(self, X, column_major=True):
        X = np.asarray(X)
        order = 'F' if column_major else 'C'
        if X.ndim < 2:
            return X
        elif X.ndim == 2:
            return X.ravel(order=order)
        return np.asarray([X[:, :, i].ravel(order=order) for i in range(X.shape[2])])

    def binarize(self, vector, term_size, min_value=0, max_value=1, mean=0):
        #vector_bin = self.flatten(self.thermometerEncoderMinMax(vector, term_size, min_value, max_value))
        vector_bin = self.flatten(self.thermometerEncoder(vector, mean))
        return vector_bin


    def get_labels(self, dataframe: pd.DataFrame, columns: np.array) -> np.array:
        #retorna uma matriz com as labels binárias para a BR. A matriz aqui é transposta. 
        #retorna um vetor com as labels. O tamanho do vetor é igual ao número de documentos.
        binary_labels = []
        for index in columns:
            binary_labels.append(dataframe[columns[index][0]].values)
        binary_labels = np.array(binary_labels)
        binary_labels = binary_labels.transpose()
        powerset_labels = []
        for label in binary_labels:
            ps = "".join([str(x) for x in label])
            powerset_labels.append(ps)
        return binary_labels, powerset_labels

    def preprocessing(self, dataset, column_name):
        documents = self.dataset[column_name]
        X_preprocessed = []
        preproc = Preprocessing()
        for text in documents:
            cleaned_text = preproc.remove_punctuations(text)
            cleaned_text = preproc.remove_stop_words(cleaned_text)
            cleaned_text = " ".join([word for word in cleaned_text])
            lemma_text = preproc.lemmatize_tokenize(self.__nlp_model, cleaned_text)
            final_text = " ".join([word for word in lemma_text])
            X_preprocessed.append(final_text)
        return X_preprocessed

    def term_frequency(self, dataset):
        vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 3))
        X_term_frequency = vectorizer2.fit_transform(dataset)
        X_term_frequency = X_term_frequency.todense()
        #self.__X_term_frequency = pd.DataFrame(self.__X_term_frequency)
        return X_term_frequency