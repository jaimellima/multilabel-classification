#experimentos com os arquivos binarizados .json
#após a binarização com o binarization.py
#usa Classifier

from classifier import Classifier
from sklearn.model_selection import train_test_split
import config as cfg
import json
import numpy as np
import pandas as pd


clf = Classifier()

df = pd.read_csv("/home/jolima/Documentos/multilabel-classification/dataset/binaries/kaggle_10/3.csv")
X = df["features"][0]
X = X.rstrip('\n')
print(X)
y = df["labels"]
print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print(X_train)
print(X_test)
print(y_train)
print(y_test)
#y_pred = clf.wisard(X_train, X_test, y_train, cfg.RAM_STD)
#print(y_pred)

