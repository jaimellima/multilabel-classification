#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:00:23 2022

@author: jolima
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import argparse
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import numpy as np
import gensim.downloader as gensim_api
from gensim.models.word2vec import Word2Vec
import gensim.models



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
    print("Getting Lemma...")
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

def tf_idf_vectorization(dataframe, column_to_fit, max_features=500, max_df=0.85):
    print("Getting TF-IDF...")
    print("Resetting indexes...")
    dataframe = dataframe.reset_index()
    vetorizer = TfidfVectorizer(max_features=max_features, max_df=max_df)
    X_tfidf = vetorizer.fit_transform(dataframe[column_to_fit])
    df_tfidf = pd.DataFrame(X_tfidf.todense())
    df_tfidf["labels"] = dataframe["all_tags"]
    return df_tfidf

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)
    print("Saving matrix {}!".format(filename))
    

#def load_sparse_csr(filename):
#    loader = np.load(filename)
#    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
#                      shape=loader['shape'])

def load_gensim(model="word2vec-google-news-300"):
    nlp = gensim_api.load("word2vec-google-news-300")
    return nlp
    
def get_unigrams(dataframe, column):
    corpus = dataframe[column]
    lst_corpus = []
    for token in corpus:
        lst_words = token.split()
        lst_grams = [" ".join(lst_words[i:i+1]) for i in range(0, len(lst_words), 1)]
        lst_corpus.append(lst_grams)
    return lst_corpus

   
    
def main():
    parser = argparse.ArgumentParser(description = 'WISARD weightless neural network preprocessor')
    
        
    parser.add_argument('--file', 
                    action='store', 
                    dest='file', 
                    default='./dataset.csv', 
                    required=True, 
                    help='A valid dataset (.csv) must be entered.')
    
    parser.add_argument('--n_sample', 
                    action='store', 
                    dest='n_sample', 
                    default=10,
                    type=int,
                    required=True, 
                    help='Number of samples to be used for training and testing.')
    
    parser.add_argument('--featureSource', 
                    action='store', 
                    dest='featureSource', 
                    default=1,
                    type=int,
                    required=True, 
                    help='0: Title. 1: Title and Abstract. 2. Only NER')
    
    arguments = parser.parse_args()
    
    file = arguments.file
    n_sample = arguments.n_sample
    featureSource = arguments.featureSource
    
   
    df_kaggle = load_csv(file, n=n_sample)
    if featureSource == 1:
        df_kaggle = concat_columns(df_kaggle, ["TITLE","ABSTRACT"], "text")
    else:
        df_kaggle = concat_columns(df_kaggle, ["TITLE"], "text")
        
    df_kaggle = zip_columns_kaggle(df_kaggle)
    df_kaggle = transform(df_kaggle, "text")
    nlp = spacy.load("en_core_web_lg")
    df_kaggle = lemmatization(df_kaggle, "text", "text_lemma", nlp)
    df_tfidf = tf_idf_vectorization(df_kaggle, 
                                   "text_lemma", 
                                   max_features=5000,
                                   max_df=0.85)
    #file_tfidf_name = "tf_idf_matrix"
    #save_sparse_csr(file_tfidf_name, X_tfidf)
    #df_tfidf = pd.DataFrame(X_tfidf.todense())
    df_tfidf.to_csv("vectors_tfidf.csv", index=False)
    print("TF-IDF DONE!!!...")   

if __name__=="__main__":
    main()

