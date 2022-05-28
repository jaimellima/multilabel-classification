
import pandas as pd
import numpy as np
import config as cfg
import en_core_web_lg
import json

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

print("Trying to acess CSV file in {}".format(cfg.FILE_PATH))
dataset = pd.read_csv(cfg.FILE_PATH)
# print(dataset)

# preprocessing = Preprocessing(path_dataset=cfg.KAGGLE_DATASET, n_sample=cfg.SAMPLE,
#      X_columns=["TITLE", "ABSTRACT"], y_columns=['Computer Science', 'Physics'],
#      column_text_name="TEXT", nlp_model=en_core_web_lg.load(), stop_words=STOP_WORDS)

# preprocessing.preprocessing()
# preprocessing.term_frequency()

# print(preprocessing.X_term_frequency.shape)

preprocessing = Preprocessing(path_dataset=cfg.FILE_PATH, 
                              n_sample=cfg.SAMPLE,
                              X_columns=["TITLE", "ABSTRACT"], 
                              y_columns=cfg.Y_COLUMNS,
                              column_text_name="TEXT", 
                              nlp_model=en_core_web_lg.load(), 
                              stop_words=STOP_WORDS)

X_preprocessed = preprocessing.preprocessing()
X_vectorized = preprocessing.vectorize(method="TF", X_preprocessed=X_preprocessed)

for term_size in range(cfg.MIN_TERM_SIZE, cfg.MAX_TERM_SIZE+1):
    d = dict()
    print("Processing binary matrix for {} value".format(term_size))
    X_bin = preprocessing.binarize(term_size=term_size, X_vectorized=X_vectorized)
    d[term_size] = X_bin.tolist()
    file_name = cfg.BINARIES_DIR + "/" + str(term_size) + ".json"
    print("Saving {}".format(file_name))
    with open(file_name, 'w') as f:
       json.dump(d, f)



#LER O ARQUIVO PARA DICIONÁRIO
#with open(cfg.JSON_BINARY) as f:
#    data = json.load(f)

#for item in data:
#    print(item)
