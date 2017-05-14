# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:30:05 2017

@author: song-46
"""

from sklearn import datasets
from sklearn import svm

clf1 = svm.SVC()
iris = datasets.load_iris()
X = iris.data
y = iris.target
clf1.fit(X, y)

# method 1: pickle
import pickle
# Save
##with open("clf2.pickle", "wb") as f:
##    pickle.dump(clf1, f)
# Restore
##with open("clf2.pickle", "rb") as f:
##    clf = pickle.load(f)
##    print(clf.predict(X[0:5]))

# method 2: joblib
from sklearn.externals import joblib
# Save
##joblib.dump(clf1, "clf3.pkl")
# Restore
##clf = joblib.load("clf3.pkl")
##print(clf.predict(X[0:5]))
