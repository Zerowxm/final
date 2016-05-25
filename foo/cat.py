# -*- coding: utf-8 -*-
"""
Created on Mon May 09 13:06:35 2016

@author: Zero
"""

from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve,auc
from statsmodels.tools import categorical
import numpy as np

iris = datasets.load_iris()
# Use only data for 2 classes.
X = iris.data[(iris.target==0) | (iris.target==1)]
Y = iris.target[(iris.target==0) | (iris.target==1)]

# Class 0 has indices 0-49. Class 1 has indices 50-99.
# Divide data into 80% training, 20% testing.
train_indices = list(range(40)) + list(range(50,90))
test_indices = list(range(40,50)) + list(range(90,100))
X_train = X[train_indices]
X_test = X[test_indices]
y_train = Y[train_indices]
y_test = Y[test_indices]


###########################################################################
###### Convert categorical variable to matrix and merge back with training
###### data.

# Fake categorical variable.
catVar = np.array(['a']*40 + ['b']*40)
catVar = categorical(catVar, drop=True)
X_train = np.concatenate((X_train, catVar), axis = 1)

catVar = np.array(['a']*10 + ['b']*10)
catVar = categorical(catVar, drop=True)
X_test = np.concatenate((X_test, catVar), axis = 1)
###########################################################################

# Model and test.
clf = GradientBoostingClassifier(learning_rate=0.01,max_depth=8,n_estimators=50).fit(X_train, y_train)

prob = clf.predict_proba(X_test)[:,1]   # Only look at P(y==1).

fpr, tpr, thresholds = roc_curve(y_test, prob)
roc_auc_prob = auc(fpr, tpr)

print(prob)
print(y_test)
print(roc_auc_prob)