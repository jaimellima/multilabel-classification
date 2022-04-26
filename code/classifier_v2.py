import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import argparse
import datetime 
import os
import wisardpkg as wsd


import spacy
from spacy.lang.en.stop_words import STOP_WORDS
                 
def load_csv(path):
    df = pd.read_csv(path)
    return df

def get_sample(df, n=100):
    return df.sample(n=n)

def thermometerEncoder(X, size, min=0, max=1):
    X = np.asarray(X)
    if X.ndim == 0:
        f = lambda i: X >= min + i*(max - min)/size
    elif X.ndim == 1:
        f = lambda i, j: X[j] >= min + i*(max - min)/size
    else:
        f = lambda i, j, k: X[k, j] >= min + i*(max - min)/size 
    return  np.fromfunction(f, (size, *reversed(X.shape)), dtype=int).astype(int)

def flatten(X, column_major=True):
    X = np.asarray(X)
    order = 'F' if column_major else 'C'
    if X.ndim < 2:
        return X
    elif X.ndim == 2:
        return X.ravel(order=order)
    return np.asarray([X[:, :, i].ravel(order=order) for i in range(X.shape[2])])

def log_reg_one_vs_rest(X_train_tfidf, X_test_tfidf, y_train, y_test):
    log_reg = LogisticRegression()
    classifier = OneVsRestClassifier(log_reg)
    classifier.fit(X_train_tfidf, y_train)
    result = classifier.score(X_test_tfidf, y_test)
    print("Score LR: {}".format(result))
    y_pred = classifier.predict(X_test_tfidf)
    hl_result = hamming_loss(y_test, y_pred)
    print("Hamming Loss LR: {}".format(hl_result))
    return y_pred

def svm_one_vs_rest(X_train_tfidf, X_test_tfidf, y_train, y_test):
    svm_ = svm.SVC()
    classifier = OneVsRestClassifier(svm_)
    classifier.fit(X_train_tfidf, y_train)
    result = classifier.score(X_test_tfidf, y_test)
    print("Score SVM: {}".format(result))
    y_pred = classifier.predict(X_test_tfidf)
    hl_result = hamming_loss(y_test, y_pred)
    print("Hamming Loss SVM: {}".format(hl_result))
    return y_pred
    
def prepare_dataset_word2vec(dataframe, column_to_bin, column_dest_name, tags, term_size, file_name):
    for index, row in dataframe.iterrows():
        vector = row[column_to_bin]
        vector = vector.replace("[", "")
        vector = vector.replace("]", "")
        vector = vector.replace("\n", "")
        vector = vector.split()
        vector = np.array(vector)
        vector = vector.astype(float)
        vector_bin = flatten(thermometerEncoder(vector, term_size, min(vector), max(vector)))
        #vector_bin = flatten(thermometerEncoder(vector, term_size, 0, max(vector)))
        print("Min.: {} Max.: {}".format(min(vector), max(vector)))
        print("Binary vector size: {}".format(len(vector_bin)))
        vector_str = ' '.join(str(i) for i in vector_bin)
        dataframe.loc[index, column_dest_name] = vector_str
    dataframe.to_csv(file_name)

def prepare_dataset_tfidf(dataframe, column_dest_name, term_size, file_name):
    print("Dataframe Shape: {}".format(dataframe.shape[1]))
    row_len = dataframe.shape[1] - 1
    print("Tamanho da linha: {}".format(row_len))
    vectors_bin = []
    
    dataframe = dataframe.reset_index()
    
    for index, row in dataframe.iterrows():
        vector = dataframe.iloc[index,0:row_len].values
        vector = vector.astype(float)
        vector_bin = flatten(thermometerEncoder(vector, term_size, min(vector), max(vector)))    
        vectors_bin.append(vector_bin)
        vector_str = ' '.join(str(i) for i in vector_bin)
        dataframe.loc[index, column_dest_name] = [vector_str]
    dataframe.to_csv(file_name)
        
def get_y_matrix(y_test, y_pred):
    y_test_matrix = []
    for item in y_test:
        x = item.split()
        y_test_matrix.append(x)
    y_pred_matrix = []
    for item in y_pred:
        x = item.split()
        y_pred_matrix.append(x)
    y_test_matrix = np.array(y_test_matrix)
    y_pred_matrix = np.array(y_pred_matrix)
    y_test_matrix = y_test_matrix.astype(int)
    y_pred_matrix = y_pred_matrix.astype(int)
    return y_test_matrix, y_pred_matrix
    
def wisard_label_powerset(X_train, y_train, X_test, y_test, ram):
    wisard = wsd.Wisard(ram, ignoreZero=False)
    wisard.train(X_train, y_train)
    y_pred = np.array(wisard.classify(X_test))
    print("Predictions - ")
    print("Total of predictions: {}".format(len(y_pred)))
    acc = accuracy_score(y_test, y_pred)
    print("Label Powerset - Accuracy: {}".format(acc))
    y_test_matrix, y_pred_matrix = get_y_matrix(y_test, y_pred)
    hl = hamming_loss(y_test_matrix, y_pred_matrix)
    print("Label Powerset - Hamming Loss: {}".format(hl))
    print()
    return acc, hl

def wisard_binary_relevance(tags, X_train, y_train, X_test, y_test, ram):
    labels_train = []
    for item in y_train:
        item_temp = np.array(item.split())
        labels_train.append(item_temp)
    labels_train = np.array(labels_train)
    labels_train = labels_train.T
    
    labels_test = []
    for item in y_test:
        item_temp = np.array(item.split())
        labels_test.append(item_temp)
    labels_test = np.array(labels_test)
    labels_test = labels_test.T
    
    acc = []
    hl = []
    
    for tag in tags:
        print("Training and Testing Wisard for {}".format(tag))
        wisard = wsd.Wisard(ram, ignoreZero=False)
        wisard.train(X_train, labels_train[tags[tag]])
        y_pred = np.array(wisard.classify(X_test))
    
        acc.append(accuracy_score(labels_test[tags[tag]], y_pred))
        hl.append(hamming_loss(labels_test[tags[tag]], y_pred))
    acc_mean = np.mean(acc)
    hl_mean = np.mean(hl)
    print("Binary Relevance (Avg) - Accuracy: {}".format(acc_mean))
    print("Binary Relevance (Avg) - Hamming Loss: {}".format(hl_mean))
    print()
    return acc_mean, hl_mean
    
    #labels_train = []
    #for tag in tags:
    #    y_train_new = []
    #    for item in y_train:
    #        y_train_temp = item.split()
    #        y_train_new.append(y_train_temp[tags[tag]])
    #    labels_train.append(y_train_new)
    #labels_test = []
    #for tag in tags:
    #    y_test_new = []
    #    for item in y_test:
    #        y_test_temp = item.split()
    #        y_test_new.append(y_test_temp[tags[tag]])
    #    labels_test.append(y_test_new)
    #wisard = wsd.Wisard(ram, ignoreZero=False)
    #wisard.train(X_train, y_train_new[0])
    #y_pred = np.array(wisard.classify(X_test))
    #print(y_pred)
    
    #for tag in tags:
    #    labels_bin.append(y_train[])
        
    
def wsd_w2v_experiment(tags, dataset, min_ter, max_ter, n_iter, min_ram, max_ram):
    for thermometer in range(min_ter, max_ter+1):
        fname_w2v = "df_binary_w2v.csv"
        print("Thermometer Parameter: {}".format(thermometer))
        prepare_dataset_word2vec(dataset, "vector", "binary", tags, thermometer, fname_w2v)
        dataset_bin = load_csv(fname_w2v)
        X = []
        y = []
        X_temp = dataset_bin["binary"]
        y_temp = dataset_bin["labels"]
        for item in X_temp:
            vec = item.split()
            vec = [int(i) for i in vec]
            X.append(vec)
        for item in y_temp:
            item = item.replace("(", "")
            item = item.replace(")", "")
            labels = item.split(",")
            labels = "".join([str(i) for i in labels])
            y.append(labels)
        #for label in y:
        #    print(label)
        for iteration in range(0, n_iter):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=iteration)
            for ram in range(min_ram, max_ram+1):
                print("RAM Parameter: {}".format(ram))
                acc, hl = wisard_label_powerset(X_train, y_train, X_test, y_test, ram)
                wisard_binary_relevance(tags_to_class, X_train, y_train, X_test, y_test, ram)
            
def wsd_tfidf_experiment(dataset, min_ter, max_ter, n_iter, min_ram, max_ram):
    results = pd.DataFrame(columns=["N_ITER","MIN_TER","MAX_TER","MIN_RAM",
                                    "MAX_RAM","RAM","THERMOMETER","ITERATION",
                                    "ACC_LP","HL_LP","ACC_BR","HL_BR"])    
    #results_fname = results
    date = datetime.datetime.now()
    results_fname = "results_{}".format(date.strftime("%y%m%d_%H%M%S"))
    print(results_fname)
    for thermometer in range(min_ter, max_ter+1):
        fname_tfidf = "df_binary_tfidf.csv"
        print("TF-IDF - Thermometer Parameter: {}".format(thermometer))
        prepare_dataset_tfidf(dataset, "binary", thermometer, fname_tfidf)
        dataset_bin = load_csv(fname_tfidf)
        X = []
        y = []
        X_temp = dataset_bin["binary"]
        y_temp = dataset_bin["labels"]
        for item in X_temp:
            vec = item.split()
            vec = [int(i) for i in vec]
            X.append(vec)
        for item in y_temp:
            item = item.replace("(", "")
            item = item.replace(")", "")
            labels = item.split(",")
            labels = "".join([str(i) for i in labels])
            y.append(labels)
        #for label in y:
        #    print(label)
        for iteration in range(0, n_iter):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=iteration)
            for ram in range(min_ram, max_ram+1):
                print("TF-IDF - RAM Parameter: {}".format(ram))
                acc_lp = 0
                hl_lp = 0
                acc_br = 0
                hl_br = 0
                acc_lp, hl_lp = wisard_label_powerset(X_train, y_train, X_test, y_test, ram)
                acc_br, hl_br = wisard_binary_relevance(tags_to_class, X_train, y_train, X_test, y_test, ram)
                new_dict = {"N_ITER": n_iter,
                            "MIN_TER": min_ter, 
                            "MAX_TER": max_ter,
                            "MIN_RAM": min_ram, 
                            "MAX_RAM": max_ram,
                            "RAM": ram,
                            "THERMOMETER": thermometer,
                            "ITERATION": iteration,
                            "ACC_LP": acc_lp,
                            "HL_LP": hl_lp,
                            "ACC_BR": acc_br,
                            "HL_BR": hl_br}
                results = results.append(new_dict, ignore_index=True)
    results.to_csv(results_fname)
                
               
               
if __name__=="__main__":
    
    #criar json com as classes
    tags_to_class = {"Computer Science": 0,
        "Physics": 1,
        "Mathematics": 2,
        "Statistics": 3,
        "Quantitative Biology": 4,
        "Quantitative Finance": 5
        } 
    
     
    parser = argparse.ArgumentParser(description = 'WISARD weightless neural network based multilabel classifier')
    parser.add_argument('--file', 
                    action='store', 
                    dest='file', 
                    default='./dataset.csv', 
                    required=True,
                    help='A valid dataset (.csv) must be entered.')
    parser.add_argument('--n_iter', 
                    action='store', 
                    dest='n_iter', 
                    default=1,
                    type=int,
                    required=False, 
                    help='Number of iterations/repetitions for the informed hyperparameters.')
    parser.add_argument('--min_ram', 
                    action='store', 
                    dest='min_ram', 
                    default=3,
                    type=int,
                    required=False, 
                    help='Minimum value for RAM. Default WISARD RAM = 3.')
    
    parser.add_argument('--max_ram', 
                    action='store', 
                    dest='max_ram', 
                    default=4, 
                    type=int,
                    required=True, 
                    help='Maximum value for RAM. Default WISARD RAM = 4.')
    
    parser.add_argument('--max_ter', 
                    action='store', 
                    dest='max_ter', 
                    default=3, 
                    type=int,
                    required=False, 
                    help='Maximum value for thermometer. Default value = 3.')
    
    parser.add_argument('--min_ter', 
                    action='store', 
                    dest='min_ter', 
                    default=4,
                    type=int,
                    required=True, 
                    help='Maximum value for thermometer. Default value = 4.')
    
    parser.add_argument('--n_sample', 
                    action='store', 
                    dest='n_sample', 
                    default=10,
                    type=int,
                    required=True, 
                    help='Number of samples to be used for training and testing.')

    parser.add_argument('--preprocess', 
                    action='store', 
                    dest='preprocess', 
                    default=1,
                    type=int,
                    required=True, 
                    help='0: Doc2Vec Spacy. 1: TF-IDF.')
    
    parser.add_argument('--method', 
                    action='store', 
                    dest='method', 
                    default=0,
                    type=int,
                    required=True, 
                    help='0: Label Powerset. 1: Binary Relevance')
    
    parser.add_argument('--featureSource', 
                    action='store', 
                    dest='featureSource', 
                    default=1,
                    type=int,
                    required=True, 
                    help='0: Title. 1: Title and Abstract. 2. Only NER')
    
    arguments = parser.parse_args()
    path_kaggle = "../dataset/kaggle_dataset.csv"
    file = arguments.file
    n_iter = arguments.n_iter
    min_ram = arguments.min_ram
    max_ram = arguments.max_ram
    min_ter = arguments.min_ter
    max_ter = arguments.max_ter
    n_sample = arguments.n_sample
    preprocess = arguments.preprocess
    method = arguments.method
    featureSource = arguments.featureSource
    
    #carrega CSV, recebendo como parâmetro o número de amostras
    #df_kaggle = load_csv(path_kaggle, n=SAMPLE)
    dataset = load_csv(file)
    dataset = get_sample(dataset, n_sample)

    if preprocess:
        wsd_w2v_experiment(tags_to_class, dataset, min_ter, max_ter, n_iter, min_ram, max_ram)
    else:
        wsd_tfidf_experiment(dataset, min_ter, max_ter, n_iter, min_ram, max_ram)
        print()
        print("Preprocess Method TF-IDF: {}".format(preprocess))
        
    
    

    


            
    
    

    

        

        
       
    
    #concatena colunas de um dataframe
 
        
    #resultado = pd.DataFrame(columns=["EXPERIMENT", "SEED", "RAM", "TERM", "HAM_LOS", "PREC", "REC", "F1"])
    #experiment = datetime.datetime.now()

    
    


