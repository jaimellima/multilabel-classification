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



experiment = {}
for term_size in range(cfg.MIN_TERM_SIZE, cfg.MAX_TERM_SIZE+1):
   clf = Classifier()
   print("Term size: {}".format(term_size))
   fname_df_features = "{}{}_documents.csv".format(cfg.KAGGLE_100, term_size)
   print(fname_df_features)
   df_features = pd.read_csv(fname_df_features)
   X = df_features.to_numpy()
   print("Documents matrix shape: {}".format(X.shape))
   file_labels = "{}{}_labels.json".format(cfg.KAGGLE_100, term_size)
   print(file_labels)
   y = None
   with open(file_labels) as f:
      data = json.load(f)
   for item in data:
      y = np.array(data[item])
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
   y_pred = clf.wisard(X_train, y_train, X_train, cfg.RAM_STD)
   experiment[term_size] = {"y_test": y_train,"y_pred": y_pred}
   


# df_features = pd.read_csv(cfg.KAGGLE_100 3_features.csv)
# file_labels = "/home/jolima/Documentos/multilabel-classification/dataset/binaries/kaggle_100/labels.json"
# X = df_features.to_numpy()
# y = None
# with open(file_labels) as f:
#    data = json.load(f)
# for item in data:
#    y = np.array(data[item])
# print(type(y))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# experiment = {}
# y_pred = clf.wisard(X_train, y_train, X_train, cfg.RAM_STD)
# print(y_pred)

# df_experiments['{}_test'.format(cfg.RAM_STD)] = y_train
# df_experiments['{}_pred'.format(cfg.RAM_STD)] = y_pred
# print(df_experiments['16_test'])
# print(df_experiments['16_pred'])

