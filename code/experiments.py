#experimentos com os arquivos binarizados .json
#após a binarização com o binarization.py
#usa Classifier

from classifier import Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import config as cfg
import json
import numpy as np
import pandas as pd
import re


clf = Classifier()

df_features = pd.read_csv("/home/jolima/Documentos/multilabel-classification/dataset/binaries/kaggle_100/3_features.csv")
file_labels = "/home/jolima/Documentos/multilabel-classification/dataset/binaries/kaggle_100/labels.json"
X = df_features.to_numpy()
y = None
with open(file_labels) as f:
   data = json.load(f)
for item in data:
   y = np.array(data[item])
print(type(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
y_pred = clf.wisard(X_train, y_train, X_train, cfg.RAM_STD)
print(y_pred)
acc = accuracy_score(y_train, y_pred)
print(acc)

