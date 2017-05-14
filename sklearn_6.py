# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:30:05 2017

@author: song-46
"""

from sklearn.learning_curve import validation_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
y = digits.target

param_range = np.logspace(-6, -2.3, 5)

train_loss, test_loss = validation_curve(
    SVC(), X, y, param_name="gamma",
    param_range=param_range, 
    cv=5, scoring="mean_squared_error")

train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(param_range, train_loss_mean, "o-", c="r", label="Training")
plt.plot(param_range, test_loss_mean, "o-", c="g", label="Cross-validation")
plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
