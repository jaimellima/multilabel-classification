# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 00:31:17 2021

@author: jaimel
"""

import pandas as pd

df = pd.read_csv("/home/jolima/Documentos/multilabel-classification/multi-label-classification/dataset/kaggle_dataset.csv")
df = df.sample(n=10)
columns = ['Computer Science', 'Physics', 'Mathematics',
       'Statistics', 'Quantitative Biology', 'Quantitative Finance']
labels = df[columns].values
for label in labels:
    ps = "".join([str(x) for x in label])
    print(ps)
    ps = list(ps)
    print(ps)

