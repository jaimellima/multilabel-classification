#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:36:31 2022

@author: jolima
"""

import numpy as np
import pandas as pd
import argparse
#comentário

print("Isso é uma alteração de teste para o github")

def load_csv(path):
    df = pd.read_csv(path)
    return df

#Função passada para arquivo de binarização
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


def main():
    parser = argparse.ArgumentParser(description = 'WISARD weightless neural network preprocessor')
    parser.add_argument('--file', 
                    action='store', 
                    dest='file', 
                    default='./dataset.csv', 
                    required=True, 
                    help='A valid dataset (.csv) must be entered.')
    
    parser.add_argument('--term_size', 
                    action='store', 
                    dest='term_size', 
                    default=16,
                    type=int,
                    required=True, 
                    help='Thermometer size for binarization process.')
    
    arguments = parser.parse_args()
    file_path = arguments.file
    term_size = arguments.term_size
    df_word2vec = load_csv(file_path)
    
    #TODO: fazer esse tratamento na vetorização, para não ter esse tipo de trabalho.
    for index, row in df_word2vec.iterrows():
        vector = row["vector"]
        vector = vector.replace("[", "")
        vector = vector.replace("]", "")
        vector = vector.replace("\n", "")
        vector = vector.split()
        vector = np.array(vector)
        vector = vector.astype(float)
        vector_bin = flatten(thermometerEncoder(vector, term_size, min(vector), max(vector)))
        print("Min.: {} Max.: {}".format(min(vector), max(vector)))
        print("Binary vector size: {}".format(len(vector_bin)))
        vector_str = ''.join(str(i) for i in vector_bin)
        df_word2vec.loc[index, "binary"] = vector_str
        
    df_word2vec.to_csv("df_word2vec_binary.csv")
    
if __name__=="__main__":
    main()