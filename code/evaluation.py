from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
import config as cfg
import numpy as np
import json

experiment_file_name = "{}experiment.json".format(cfg.KAGGLE_100)
print(experiment_file_name)

experiment_data = []
ram = None
term = None
seed = None
y_test = None
y_pred = None


with open(experiment_file_name) as f:
   experiment_file = json.load(f)
for item in experiment_file:
   y_test_matrix = []
   y_pred_matrix = []
   ram = experiment_file[item]['ram']
   term = experiment_file[item]['term']
   seed = experiment_file[item]['seed']
   y_test = experiment_file[item]['y-test']
   y_pred = experiment_file[item]['y-pred']
   print(ram, term, seed, len(y_test), len(y_pred))
   acc = accuracy_score(y_test, y_pred)
   for label in y_test:
        y_test_matrix.append([int(i) for i in label])
   y_test_matrix = np.array(y_test_matrix)
   for label in y_pred:
        y_pred_matrix.append([int(i) for i in label])
   y_pred_matrix = np.array(y_pred_matrix)
   hl = hamming_loss(y_test_matrix, y_pred_matrix)
   print(ram, term, seed, acc, hl)


