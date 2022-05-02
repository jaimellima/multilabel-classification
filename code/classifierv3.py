import re
from sklearn.metrics import accuracy_score
import spacy
import en_core_web_lg
import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import seaborn as sns; sns.set()
import wisardpkg as wsd

from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
from distutils.command.clean import clean
from sklearn.metrics import hamming_loss


class Preprocessing:
    
    def __init__(self):
        self.stop_words=STOP_WORDS
  
    def remove_punctuations(self, text):
        text = text.lower()
        cleaned_text = re.findall("[a-zA-Z]+", text)
        return cleaned_text

    def remove_stop_words(self, text):
        cleaned_text = [word for word in text if not word in self.stop_words]
        return cleaned_text

    def lemmatize_tokenize(self, text, nlp_model):
        lemmas = []
        doc = nlp_model(text)
        for token in doc:
            if not token.is_stop:
                lemmas.append(token.lemma_)
        return lemmas

    def tf_idf_vectorization(self, documents, max_features=500, max_df=0.85):
        #recebe um vetor/conjunto de documento e retorna uma matriz TF-IDF
        print("Getting TF-IDF...")
        print("Resetting indexes...")
        vetorizer = TfidfVectorizer(max_features=max_features, max_df=max_df)
        X_tfidf = vetorizer.fit_transform(documents)
        X_tfidf = X_tfidf.todense()
        return X_tfidf

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
    def thermometerEncoderMean(self,X, size, mean):
        pass

    #cria o flatten a partir do dados binarizados pelo termômetro.
    def flatten(self, X, column_major=True):
        X = np.asarray(X)
        order = 'F' if column_major else 'C'
        if X.ndim < 2:
            return X
        elif X.ndim == 2:
            return X.ravel(order=order)
        return np.asarray([X[:, :, i].ravel(order=order) for i in range(X.shape[2])])

    def binarize(self, vector, term_size, min_value, max_value):
        vector_bin = self.flatten(self.thermometerEncoderMinMax(vector, term_size, min_value, max_value))
        print(vector_bin[0])
        return vector_bin[0]

    def get_labels(self, dataframe, columns):
        #retorna uma matriz com as labels binárias para a BR. A matriz aqui é transposta. 
        #retorna um vetor com as labels. O tamanho do vetor é igual ao número de documentos.
        binary_labels = dataframe[columns].values
        powerset_labels = []
        for label in binary_labels:
            ps = "".join([str(x) for x in label])
            powerset_labels.append(ps)
        binary_labels = binary_labels.transpose()
        return binary_labels, powerset_labels

class Classifing:
    def __init__(self, classifier):
        self.classifier=classifier

    def wisard_label_powerset(self, X_train, y_train, X_test, y_test, ram):
        wisard = wsd.Wisard(ram, ignoreZero=False)
        wisard.train(X_train, y_train)
        y_pred = np.array(wisard.classify(X_test))
        acc = accuracy_score(y_test, y_pred)
        return acc, y_pred

    def wisard_binary_relevance(self, X_train, y_train, X_test, y_test, ram, tags):
        pass
        
if __name__=="__main__":
    #documents = ["It is a, test of a Text tested.[];;", "It is a second beautiful document for test"]
    path_dataset = "../dataset/kaggle_dataset.csv"
    term_size = 16
    dataframe = pd.read_csv(path_dataset)
    dataframe["TEXT"] = dataframe["TITLE"] + dataframe["ABSTRACT"] 
    dataframe = dataframe.sample(n=100)
    documents = dataframe["TEXT"].values
    processed_documents = []
    prep = Preprocessing()
    nlp = en_core_web_lg.load()
    for text in documents:
        processed_text = prep.remove_punctuations(text)
        processed_text = prep.remove_stop_words(processed_text)
        processed_text = " ".join([word for word in processed_text])
        processed_text = prep.lemmatize_tokenize(processed_text, nlp)
        processed_text = " ".join([word for word in processed_text])
        processed_documents.append(processed_text)
    #print(processed_documents)
    vectors = prep.tf_idf_vectorization(processed_documents, max_features=5000)
    #vectors = prep.doc2vec_spacy(processed_documents, nlp)
    vectors_bin = []
    for vector in vectors:
        v = prep.binarize(vector, term_size,  np.min(vector), np.max(vector))
        vectors_bin.append(v)
    vectors_bin = pd.DataFrame(vectors_bin)
    columns = ['Computer Science', 'Physics', 'Mathematics',
       'Statistics', 'Quantitative Biology', 'Quantitative Finance']
    y_br, y_ps = prep.get_labels(dataframe, columns)
    classifier = Classifing(0)
    #y_br_0 = np.array(y_br[0].astype(str))
    #print(y_br_0)
    print(vectors_bin)
    print(y_ps)
    acc, y_pred= classifier.wisard_label_powerset(vectors_bin, y_ps, vectors_bin, y_ps, 32)
    hl = hamming_loss(y_ps, y_pred)
    print(acc)
    print(hl)
    print(y_pred)
    #Marcos falou que TF é melhor sim que TF-IDF

    
