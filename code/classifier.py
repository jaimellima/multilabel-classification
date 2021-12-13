import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


import wisardpkg as wsd

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
                 
def load_csv(path, n=100):
    df = pd.read_csv(path)
    return df.sample(n=n)

def concat_columns(dataframe, columns, new_column):
    #INPUT: dataframe, columns:list of str, new_column: str
    #CONCATENA OS TEXTOS DAS COLUNAS PASSADAS COMO PARÂMENTROS
    #OUTPUT: dataframe[new_column]
    dataframe[new_column] = ""
    for column in columns:
        dataframe[new_column] = dataframe[new_column] + " " + dataframe[column]
    return dataframe

def transform(dataframe, column):
    dataframe[column] = dataframe[column].str.lower()
    dataframe[column] = dataframe[column].str.replace('\W',' ')
    return dataframe
    
def zip_columns_kaggle(dataframe):
    lista_zip_tags = list(zip(dataframe["Computer Science"],
                     dataframe["Physics"],
                     dataframe["Mathematics"],
                     dataframe["Statistics"],
                     dataframe["Quantitative Biology"],
                     dataframe["Quantitative Finance"]))
    dataframe["all_tags"] = lista_zip_tags
    return dataframe
    
def lemmatization(dataframe, column_to_lemma, new_column, nlp):
    dataframe[new_column] = ""
    for index, row in dataframe.iterrows():
        doc = nlp(row[column_to_lemma])
        words = list()
        text_to_insert = ""
        for token in doc:
            if not token.is_stop:
                words.append(token.lemma_)
                text_to_insert = " ".join(words) 
        dataframe.loc[index, new_column] = text_to_insert
    return dataframe

def tf_idf_vectorization(dataframe, column_to_fit, X_train, y_train, max_features=5000, max_df=0.85):
    vetorizer = TfidfVectorizer(max_features=max_features, max_df=max_df)
    vetorizer.fit(dataframe[column_to_fit])
    X_train_tfidf = vetorizer.transform(X_train)
    X_test_tfidf = vetorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf
    

def split(dataframe, X_cols, y_cols, test_size=0.33, random_state=123):
    X_train, X_test, y_train, y_test = train_test_split(
    dataframe[X_cols],
    dataframe[y_cols],
    test_size = test_size,
    random_state = random_state)
    return X_train, X_test, y_train, y_test

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
    
def wisard(X_train_tfidf_bin, y_train, X_test_tfidf_bin, y_test, num):
    wisard = wsd.Wisard(num, ignoreZero=False)
    wisard.train(X_train_tfidf_bin, y_train)
    y_pred = np.array(wisard.classify(X_test_tfidf_bin))
    return y_pred

def wisard_classifier(tags_to_class, X_train_tfidf, y_train, X_test_tfidf, y_test, num, size):
    X_train_tfidf_bin = flatten(thermometerEncoder(X_train_tfidf, 
                                                   min = np.min(X_train_tfidf), 
                                                   max=np.max(X_train_tfidf), size=size))
    X_test_tfidf_bin = flatten(thermometerEncoder(X_test_tfidf, 
                                                   min = np.min(X_test_tfidf), 
                                                   max=np.max(X_test_tfidf), size=size))
    
    pred_matrix = list()
    for tag in tags_to_class:
        #print("Training WISARD for {}".format(tag))
        y_train_ = y_train[:,tags_to_class[tag]].astype(str)
        y_test_ = y_test[:,tags_to_class[tag]].astype(str)
        y_pred =  wisard(X_train_tfidf_bin, y_train_, X_test_tfidf_bin, y_test_, num=num)
        pred_matrix.append(y_pred)
    pred_matrix = np.array(pred_matrix)
    pred_matrix = pred_matrix.astype(np.int32)
    pred_matrix = pred_matrix.T
    hl_result = hamming_loss(y_test, pred_matrix)
    print("Hamming Loss WSD RAM {}-TERM {}: {}".format(num, size, hl_result))
    return pred_matrix


    
if __name__=="__main__":
    path_kaggle = "../dataset/kaggle_dataset.csv"
    tags_to_class = {"Computer Science": 0,
        "Physics": 1,
        "Mathematics": 2,
        "Statistics": 3,
        "Quantitative Biology": 4,
        "Quantitative Finance": 5
        }
    N_SEEDS = 2
    MIN_RAM = 3
    MAX_RAM = 100
    MIN_TER = 3
    MAX_TER = 100
    SAMPLE = 10
    #carrega CSV, recebendo como parâmetro o número de amostras
    df_kaggle = load_csv(path_kaggle, n=SAMPLE)
    #concatena colunas de um dataframe
    df_kaggle = concat_columns(df_kaggle, ["TITLE","ABSTRACT"], "text")
    df_kaggle = zip_columns_kaggle(df_kaggle)
    df_kaggle = transform(df_kaggle, "text")
    nlp = spacy.load("en_core_web_sm")
    df_kaggle = lemmatization(df_kaggle, "text", "text_lemma", nlp)
    
        
    resultado = pd.DataFrame(columns=["SEED", "RAM", "TERM", "HAM_LOS", "PREC", "REC", "F1"])
    for seed in range(0, N_SEEDS):
        print("Caculado experimentos para seed: {}".format(seed))
        X_train, X_test, y_train, y_test = split(df_kaggle, "text", "all_tags", random_state=seed)    
        X_train_tfidf, X_test_tfidf = tf_idf_vectorization(df_kaggle, 
                                                           "text_lemma", 
                                                           X_train, 
                                                           X_test, 
                                                           max_features=5000,
                                                           max_df=0.85)
        
        y_train = np.asarray(list(y_train))
        y_test = np.asarray(list(y_test))
        X_train_tfidf = X_train_tfidf.todense()
        X_test_tfidf = X_test_tfidf.todense()
        for ram in range(MIN_RAM, MAX_RAM+1):
            for size_ter in range(MIN_TER, MAX_TER+1):
                pred_matrix = wisard_classifier(tags_to_class, X_train_tfidf, 
                                    y_train, X_test_tfidf, 
                                    y_test, num=ram, size=size_ter)
                hl_result = hamming_loss(y_test, pred_matrix)
                p_result = precision_score(y_test, pred_matrix, pos_label='positive', average='micro')
                r_result = recall_score(y_test, pred_matrix, pos_label='positive', average='micro')
                f_result = f1_score(y_test, pred_matrix, pos_label='positive', average='micro')
                print(hl_result)
                print(p_result)
                print(r_result)
                print(f_result)
                print(seed)
                print(ram)
                print(size_ter)
                resultado.loc[-1] = [seed, ram, size_ter, hl_result, p_result, r_result, f_result]
                resultado.index = resultado.index + 1
                resultado = resultado.sort_index()
    
        resultado.to_csv("resultados_wisard_ram_{}_{}_term_{}_{}_amostra_{}_seed_{}.csv".format(
            MIN_RAM, 
            MAX_RAM,
            MIN_TER,
            MAX_TER,
            SAMPLE,
            seed))
    
    
    
# =============================================================================
#     y_pred = log_reg_one_vs_rest(X_train_tfidf, X_test_tfidf, 
#                                  y_train, y_test)
#     
#     y_pred = svm_one_vs_rest(X_train_tfidf, X_test_tfidf, 
#                              y_train, y_test)
#     pred_matrix = wisard_classifier(tags_to_class, X_train_tfidf, 
#                                     y_train, X_test_tfidf, 
#                                     y_test, num=32, size=32)
# =============================================================================
    
    

        
        
    #pred_matrix = pred_matrix.astype(np.int32)
    #hl = hamming_loss(y_test, pred_matrix)


