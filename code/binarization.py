#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:36:31 2022

@author: jolima
"""

import numpy as np

#Função passada para arquivo de binarização
#def thermometerEncoder(X, size, min=0, max=1):
#    X = np.asarray(X)
#    if X.ndim == 0:
#        f = lambda i: X >= min + i*(max - min)/size
#    elif X.ndim == 1:
#        f = lambda i, j: X[j] >= min + i*(max - min)/size
#    else:
#        f = lambda i, j, k: X[k, j] >= min + i*(max - min)/size 
#    return  np.fromfunction(f, (size, *reversed(X.shape)), dtype=int).astype(int)

#def flatten(X, column_major=True):
#    X = np.asarray(X)
#    order = 'F' if column_major else 'C'
#
#    if X.ndim < 2:
#        return X
#    elif X.ndim == 2:
#        return X.ravel(order=order)
#
#    return np.asarray([X[:, :, i].ravel(order=order) for i in range(X.shape[2])])