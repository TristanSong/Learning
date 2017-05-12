#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:17:51 2017

@author: song-46
"""

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3)
model = LinearRegression()
model.fit(X_train, y_train)

#print(model.predict(X_test))
#print(y_test)
#plt.scatter(x=range(len(X_test)), y=model.predict(X_test)-y_test)

# model's property
print(model.coef_)
print(model.intercept_)
print(model.get_params)
print(model.score(X_test, y_test))

"""
X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=0)
plt.scatter(X, y)
plt.show()
"""