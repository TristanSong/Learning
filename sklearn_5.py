# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:30:05 2017

@author: song-46
"""

from sklearn.learning_curve import learning_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
y = digits.target
train_sizes, train_loss, test_loss = learning_curve(SVC(gamma=0.001), X, y, cv=5, 
                                                    scoring="mean_squared_error", train_sizes=[0.1, 0.25, 0.5, 0.75, 1])
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(train_sizes, train_loss_mean, "o-", c="r", label="Training")
plt.plot(train_sizes, test_loss_mean, "o-", c="g", label="Cross-validation")
plt.legend("best")
plt.show()