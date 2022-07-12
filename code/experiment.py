#experimentos com os arquivos binarizados .json
#após a binarização com o binarization.py
#usa Classifier

from classifier import Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import date
import config as cfg
import json
import numpy as np
import pandas as pd
import re
from threading import Thread

class Experiment(Thread):
   def __init__(self, ram):
      Thread.__init__(self)
      self.__today = date.today()
      self.__experiment = {}
      self.__ram = ram

   def experiment(self, seed, ram, term_size):
      print("Therm size: {}".format(term_size))
      fname_df_features = "{}{}_documents.csv".format(cfg.KAGGLE_100, term_size)
      print(fname_df_features)
      df_features = pd.read_csv(fname_df_features)
      X = df_features.to_numpy()
      print("Documents matrix shape: {}".format(X.shape))
      file_labels = "{}labels.json".format(cfg.KAGGLE_100)
      print(file_labels)
      y = None
      with open(file_labels) as f:
         data = json.load(f)
      for item in data:
         y = np.array(data[item])
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = seed)
      clf = Classifier()
      y_pred = clf.wisard(X_train, y_train, X_test, ram)
      clf = Classifier()
      y_train_pred = clf.wisard(X_train, y_train, X_train, ram)
      index = "e-{}-{}-{}-{}".format(ram, term_size, seed, self.__today)
      self.__experiment[index] = {"ram": ram, "term": term_size, "seed": seed, "y-test": list(y_test),"y-pred": list(y_pred), "y-train": list(y_train),"y-train-pred": list(y_train_pred)}

   def run(self):
      print("Starting thread...")
      for term_size in range(cfg.MIN_TERM_SIZE, cfg.MAX_TERM_SIZE+1):
         for seed in range(0, cfg.SEEDS):
            print("Starting ram training and test experiment")
            self.experiment(seed, self.__ram, term_size)
      
      experiment_file_name = "{}ram_{}_exp.json".format(cfg.KAGGLE_100, self.__ram)
      print("Saving label file {}".format(experiment_file_name))
      with open(experiment_file_name, 'w') as f:
         json.dump(self.__experiment, f)
      print("Saving label file {}".format(experiment_file_name))

if __name__=="__main__":
   for ram in range(cfg.MIN_RAM_SIZE, cfg.MAX_RAM_SIZE+1):
      experiment = Experiment(ram)
      experiment.start()
