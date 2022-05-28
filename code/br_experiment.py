import pandas as pd
import numpy as np
import config as cfg
import en_core_web_lg

from preprocessing import Preprocessing
from classifier import Classifier
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss

def get_binary_matrix(y):
    y_binary = []
    for label in y:
        l = []
        for i in range(0, len(label)):
            l.append(label[i])
        y_binary.append(l)
    y_binary = np.asarray(y_binary, dtype=int)
    return y_binary

dataset = pd.read_csv(cfg.KAGGLE_DATASET)

preprocessing = Preprocessing(path_dataset=cfg.KAGGLE_DATASET, 
                              n_sample=cfg.SAMPLE,
                              X_columns=["TITLE", "ABSTRACT"], 
                              y_columns=cfg.Y_COLUMNS,
                              column_text_name="TEXT", 
                              nlp_model=en_core_web_lg.load(), 
                              stop_words=STOP_WORDS)

X_preprocessed = preprocessing.preprocessing()
X_vectorized = preprocessing.vectorize(method="TF", X_preprocessed=X_preprocessed)


#BINARY RELAVANCE EXPERIMENT
y_binary = preprocessing.binary_y
y_powerset = preprocessing.powerset_y

#TODO:INSERIR VARIAÇÃO DO TERMÔMETRO
X_bin = preprocessing.binarize(term_size=16, X_vectorized=X_vectorized)

X_train, X_test, y_train, y_test = train_test_split(X_bin, y_powerset, test_size=0.33)

indexes = preprocessing.selectKBest(X_train, y_train, method="chi2", k=500)
print("Original X train shape {}".format(X_train.shape))
X_train_ = preprocessing.filter_features(X_train, indexes)
X_test_ = preprocessing.filter_features(X_test, indexes)
print("Feature-selected X train shape {}".format(X_train_.shape))
y_binary_test = get_binary_matrix(y_test)
y_binary_train = get_binary_matrix(y_train)

results = []
for i in range(0, len(cfg.Y_COLUMNS)):
    classifier = Classifier()
    y = [label[i] for label in y_train]
    y_pred = classifier.wisard(X_train=X_train, y_train=y, X_test=X_test, ram=32)
    results.append(y_pred)

results = np.asarray(results)
y_binary_pred = []
y_powerset_pred = []
for i in range(0, results.shape[1]):
    y_binary_pred.append(results[:,i])
    y_powerset_pred.append("".join(results[:,i]))
y_binary_pred = np.asarray(y_binary_pred, dtype=int)

acc = accuracy_score(y_test, y_powerset_pred)
print("Accuracy score FS: {}".format(acc))
prec = precision_score(y_test, y_powerset_pred, average='macro')
print("Precision score FS: {}".format(prec))
rec = recall_score(y_test, y_powerset_pred, average='macro')
print("Recall score FS: {}".format(rec))
hl = hamming_loss(y_binary_test, y_binary_pred)
print("Hamming Loss FS: {}".format(hl))


