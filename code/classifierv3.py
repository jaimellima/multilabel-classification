import re
from telnetlib import X3PAD
from sklearn.metrics import accuracy_score
import spacy
import en_core_web_lg
#spacy.load("en_core_web_lg")
import pandas as pd
import numpy as np

#import matplotlib.pylab as plt
#import seaborn as sns; sns.set()
import wisardpkg as wsd

from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
from distutils.command.clean import clean
from sklearn.metrics import hamming_loss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from gensim.models import LsiModel
from gensim.test.utils import common_dictionary, common_corpus
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix


#importação do arquivo de configurações
import config as cfg


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

class FeatureSelection:

    #TODO: Aumentar o número de ocorrências em palavras que ocorrem no título e no resumo (Agrawal, 2013)
    #TODO: Implementar X² e Information Gain
    #TODO: “A typical way to output a list of selected features is ensuring a score threshold or a fixed number of features across the rankings (e.g., the top 500 features) (Pereira, 2018, p.65)”
    
    def __init__(self):
        print("Iniciando seleção de features")

    def get_features_index(self, dataset, vector_labels, label_to_select, min_tf_threshold, max_tf_threshold):
        docs_index = [ix for ix, f in enumerate(vector_labels) if f == label_to_select]
        docs_dataset = dataset[docs_index,:]
        docs_select_min = np.argwhere(docs_dataset >= min_tf_threshold)
        docs_select_min = docs_select_min[:,1]
        docs_select_max = np.argwhere(docs_dataset <= max_tf_threshold)
        docs_select_max = docs_select_max[:,1]
        features_index = np.intersect1d(docs_select_min, docs_select_max)
        return features_index

    def remove_features_by_ndocs(self, X: np.matrix, n_docs: int)-> np.array:
        """Removes features that do not occur in a number less than n_docs

        Args:
            n_docs (int): número de documentos

        Returns:
            features indexes
        """
        #retorna a quantide de documentos em cada coluna
        features = (X > 0).sum(0)
        features = [ix for ix, sum_value in enumerate(np.ravel(features)) if sum_value >= n_docs]
        return features

class Classifing:
    #TO-DO: implementar OneVsRest.
    #TO-DO: implementar ML-KNN para comparação dos resultados.

    def __init__(self, classifier):
        self.classifier=classifier

    def wisard_label_powerset(self, X_train, X_test, y_train, y_test, ram, ignore_zero=False):
        wisard = wsd.Wisard(ram, ignoreZero=ignore_zero)
        wisard.train(X_train, y_train)
        y_pred = np.array(wisard.classify(X_test))
        acc = accuracy_score(y_test, y_pred)
        return acc, y_pred

    def wisard_binary_relevance(self, X_train, y_train, X_test, y_test, ram, tags):
        pass
        
if __name__=="__main__":
    #documents = ["It is a, test of a Text tested.[];;", "It is a second beautiful document for test"]
    path_dataset = cfg.FILE_PATH
    term_size = cfg.TERM_SIZE_STD
    dataframe = pd.read_csv(path_dataset)
    dataframe["TEXT"] = dataframe["TITLE"] + dataframe["ABSTRACT"] 
    dataframe = dataframe.sample(n=cfg.SAMPLE)
    documents = dataframe["TEXT"].values
    processed_documents = []
    prep = Preprocessing()
    #nlp = spacy.load("en_core_web_lg")
    nlp = en_core_web_lg.load()
    for text in documents:
        processed_text = prep.remove_punctuations(text)
        processed_text = prep.remove_stop_words(processed_text)
        processed_text = " ".join([word for word in processed_text])
        lemma_text = prep.lemmatize_tokenize(processed_text, nlp)
        final_text = " ".join([word for word in lemma_text])
        processed_documents.append(final_text)
    print("Number of documents: {}".format(len(processed_documents)))
    columns_tag = cfg.TAGS_COLUMNS
    y_br, y_ps = prep.get_labels(dataframe, columns_tag)
    y_br_0 = np.array(y_br[:,0].astype(str))
    y_br_1 = np.array(y_br[:,1].astype(str)) 
    y_br_2 = np.array(y_br[:,2].astype(str)) 
    y_br_3 = np.array(y_br[:,3].astype(str)) 
    y_br_4 = np.array(y_br[:,4].astype(str)) 
    y_br_5 = np.array(y_br[:,5].astype(str))

    vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 3))
    X2 = vectorizer2.fit_transform(processed_documents)
    X2 = X2.todense()

    #X2 = prep.tf_idf_vectorization(processed_documents)

    label_to_select = '1'
    print("Original X Shape: ", X2.shape)
    fs = FeatureSelection()
    features_cs_index = fs.get_features_index(X2, y_br_0, label_to_select, 0.05, 0.65)
    print("len y_br_0: {}".format(len(y_br_0)))
    print("features_cs_index: {}".format(len(features_cs_index)))
    features_by_doc = fs.remove_features_by_ndocs(X2, 3)
    print("features_by_doc: {}".format(len(features_by_doc)))
    features = [feature for feature in features_cs_index if feature in features_by_doc]
    #index_to_train_0 = np.concatenate((features_cs_index, features_by_doc))
    #index_to_train_0 = np.unique(index_to_train_0)
    print()
    print("len features: {}".format(len(features)))
    print("Final X Shape: ", X2.shape)
    X2_ = X2[:,features]
    print("X_ Shape: {}".format(X2_.shape))

    X3 = SelectKBest(chi2, k=300).fit_transform(X2_, y_br_0)

    #svd = TruncatedSVD(n_components=cfg.N_COMPONENTES_SVD, n_iter=50, random_state=0)
    #svd.fit(X2_)
    #X3 = svd.transform(X2_)
    
    matriz_bin = []
    #for vector in X2_:
    #for vector in X3:
    for vector in X3:
        v = prep.binarize(vector, term_size,  np.min(vector), np.max(vector), mean=np.mean(vector))
        matriz_bin.append(v)accuracy_score
        #matriz_bin.append(v)
    #matriz_bin = np.array(matriz_bin)

    #matriz_bin_ = SelectKBest(chi2, k=200).fit_transform(matriz_bin, y_br_0)
    #print("matriz_bin_ {}".format(matriz_bin_.shape))

    X_train, X_test, y_train, y_test = train_test_split(matriz_bin, y_br_0, test_size=0.35)

    print("len y_br_0: {} {}".format(len(y_br_0), type(y_br_0)))
    #TODO: TESTAR UTILIZANDO OUTROS CLASSIFICADORES ANTES DA WISARD
    #classifier = Classifing(0)
    #acc, y_pred= classifier.wisard_label_powerset(matriz_bin, y_br_0, matriz_bin, y_br_0, cfg.RAM_STD, ignore_zero=cfg.IGNORE_ZERO_WSD)
    #print("Acurácia WiSARD Treino: {}".format(acc))

    # X_train = list(np.array(X_train, dtype=int))
    # X_test = list(np.array(X_test, dtype=int))

    # print(type(X_train))
    # print(X_train)
    # classifier = Classifing(0)
    # acc, y_pred= classifier.wisard_label_powerset(X_train, y_train, X_test, y_test, cfg.RAM_STD, ignore_zero=cfg.IGNORE_ZERO_WSD)
    # print("Acurácia WiSARD Teste: {}".format(acc))


    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Acurácia Regressão Logística: {}".format(acc))

    # classifier = MLkNN(k=3)
    # classifier.fit(X2_, y_br_0)
    # predictions = classifier.predict(X2_)
    # print(predictions)

    # clf = LogisticRegression(random_state=0).fit(artigos_treino, tags_treino)
    # y_pred = clf.predict(artigos_teste)
    # acc = accuracy_score(tags_teste, y_pred)
    # hamming_loss_onevsrest = hamming_loss(tags_teste, y_pred)
    # print(hamming_loss_onevsrest)
    # print(acc)

    # classificador_onevsrest = OneVsRestClassifier(LogisticRegression())
    # classificador_onevsrest.fit(X2_, y_br)
    # y_pred = classificador_onevsrest.predict(X2_)
    # hamming_loss_onevsrest = hamming_loss(y_br, y_pred)
    # print(hamming_loss_onevsrest)

    # classificador_onevsrest = OneVsRestClassifier(wsd.Wisard(8))
    # print(type(classificador_onevsrest))
    # classificador_onevsrest.fit(X2_, y_br)
    # y_pred = classificador_onevsrest.predict(X2_)
    # hamming_loss_onevsrest = hamming_loss(y_br, y_pred)
    # print(hamming_loss_onevsrest)

