#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:13:03 2022

@author: jolima
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import numpy as np
import gensim.downloader as gensim_api
from gensim.models.word2vec import Word2Vec
import gensim.models
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile



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

def save_model(file_name, model):
    fname = get_tmpfile("doc2vec_model")
    model.save(fname)
      
def load_gensim(model="word2vec-google-news-300"):
    nlp = gensim_api.load("word2vec-google-news-300")
    return nlp

def doc2vec_spacy(dataframe, nlp_model):
    vectors = []
    labels = []
    #TODO: Tentar mudar tamanho vetor
    for index, row in dataframe.iterrows():
        document = nlp_model(row['text_lemma'])
        vectors.append(document.vector)
        labels.append(row['all_tags'])
    return vectors, labels

def docvec_to_csv(fname, vectors, labels):
    dataframe = pd.DataFrame({"vector": vectors, "labels": labels})
    dataframe.to_csv(fname, index=False)
    #f = open(fname, "w")
    #for vector in vectors:
    #    f.write(str(vector) + "\n")
    #f.close()

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
    arguments = parser.parse_args()
    file = arguments.file
    n_sample = arguments.n_sample
    nlp = spacy.load("en_core_web_sm")
    df_kaggle = load_csv(file, n=n_sample)
    df_kaggle = concat_columns(df_kaggle, ["TITLE","ABSTRACT"], "text")
    df_kaggle = zip_columns_kaggle(df_kaggle)
    df_kaggle = transform(df_kaggle, "text")
    nlp = spacy.load("en_core_web_sm")
    df_kaggle = lemmatization(df_kaggle, "text", "text_lemma", nlp) 
    #model = gensim_api.load("word2vec-google-news-300")
    #documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df_kaggle["text_lemma"])]
    #model = Word2Vec(sentences=documents, vector_size=100, window=5, min_count=1, workers=4)
    vectors, labels = doc2vec_spacy(df_kaggle, nlp)   
    docvec_to_csv("vectors_doc2vec_spacy.csv", vectors, labels)
    
if __name__=="__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
