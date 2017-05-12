#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:11:01 2017

@author: song-46
"""

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
#Support Vector Machine中Support Vector Classifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

X, y = make_classification(
        n_samples=300, n_features=2, 
        n_redundant=0, n_informative=2, 
        random_state=22, n_clusters_per_class=1, 
        scale=100)
#plt.scatter(X[:,0], X[:,1], c=y)
#plt.show()
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))