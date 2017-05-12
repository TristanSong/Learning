#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:17:51 2017

@author: song-46
"""

import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score # K折交叉验证模块
import matplotlib.pyplot as plt

iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

#print(iris_X[:2, :])
#print(iris_Y)

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_Y, random_state=4)

#knn = KNeighborsClassifier(n_neighbors=5)
#knn.fit(X_train, y_train)
#print(knn.score(X_test, y_test))

"""
#使用K折交叉验证模块
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, iris_X, iris_Y, cv=5, scoring="accuracy")
print(scores.mean())
"""

#选择模型参数
k_scores = []
k_loss = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    loss = -cross_val_score(knn, iris_X, iris_Y, cv=5, scoring="mean_squared_error")#for regression
    score = cross_val_score(knn, iris_X, iris_Y, cv=5, scoring="accuracy")#for classification
    k_scores.append(score.mean())
    k_loss.append(loss.mean())
#plt.plot(range(1, 20), k_scores)
plt.plot(range(1, 20), k_loss)
plt.xlabel("Value of K for KNN")
plt.ylabel("Cross-validation Accuray")