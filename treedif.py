# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:18:18 2016

@author: Zero
"""

from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
 
#load data: digits.data and digits.target,
#array of features and labels, resp.
digits = load_digits(n_class =10)  
 
n_train  = []
t1_accuracy = []
t2_accuracy = []
t3_accuracy = []
 
#below, we average over "trials" num of fits for each sample
#size in order to estimate the average generalization error.
trials = 25
 
clf  = DecisionTreeClassifier()
clf2 = GradientBoostingClassifier(max_depth=3)
clf3 = RandomForestClassifier()
 
num_test = 500
 
#loop over different training set sizes
for num_train in range(2,len(digits.target)-num_test,25):
 
    acc1, acc2, acc3 = 0,0,0
 
    for j in range(trials):
        perm = [0]
        while len(set(digits.target[perm[:num_train]]))<2:
            perm = np.random.permutation(len(digits.data))
 
        clf = clf.fit(digits.data[perm[:num_train]], \
              digits.target[perm[:num_train]])
        acc1 += clf.score(digits.data[perm[-num_test:]], \
              digits.target[perm[-num_test:]])
 
        clf2 = clf2.fit(digits.data[perm[:num_train]],\
               digits.target[perm[:num_train]])
        acc2 += clf2.score(digits.data[perm[-num_test:]],\
               digits.target[perm[-num_test:]])
 
        clf3 = clf3.fit(digits.data[perm[:num_train]],\
               digits.target[perm[:num_train]])
        acc3 += clf3.score(digits.data[perm[-num_test:]],\
                digits.target[perm[-num_test:]])
 
    n_train.append(num_train)
    t1_accuracy.append(acc1/trials)
    t2_accuracy.append(acc2/trials)
    t3_accuracy.append(acc3/trials)
 
import matplotlib.pyplot as plt 
plt.plot(n_train,t1_accuracy, color = 'red')
plt.plot(n_train,t2_accuracy, color = 'green')
plt.plot(n_train,t3_accuracy, color = 'blue')

import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import cross_validation
from sklearn.datasets import load_digits

# getting digit dataset, cross validation init.
i = random.randint(1,100000)
digits = load_digits(n_class=10)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(digits.data, digits.target, test_size=0.4, random_state=i)

# Decision tree classifier.
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# Bagging classifier, 
bclf = BaggingClassifier()
bclf = bclf.fit(X_train, y_train)
bclf.score(X_test, y_test)