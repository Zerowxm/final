# -*- coding: utf-8 -*-
"""
Created on Mon May 16 22:49:58 2016

@author: Zero
"""

from __future__ import division
import pandas as pd
import numpy as np
from collections import Counter
churn_df = pd.read_csv('churn.csv')
col_names = churn_df.columns.tolist()

print "Column names:"
print col_names

to_show = col_names[:6] + col_names[-6:]

print "\nSample data:"
churn_df[to_show].head(6)
# Isolate target data
churn_result = churn_df['Churn?']
y = np.where(churn_result == 'True.',1,0)

# We don't need these columns
to_drop = ['State','Area Code','Phone','Churn?']
churn_feat_space = churn_df.drop(to_drop,axis=1)

# 'yes'/'no' has to be converted to boolean values
# NumPy converts these from boolean to 1. and 0. later
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'

# Pull out features for future use
features = churn_feat_space.columns

X = churn_feat_space.as_matrix().astype(np.float)

# This is important
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

print "Feature space holds %d observations and %d features" % X.shape
print "Unique target labels:", np.unique(y)

from sklearn.cross_validation import KFold

def run_cv(X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=5,shuffle=True)
    y_pred = y.copy()
    
    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
#        print test_index
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
#        print clf.score(X_test,y[test_index])
    return y_pred      
    
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN

def accuracy(y_true,y_pred):
    # NumPy interprets True and False as 1. and 0.
#    print y_pred,y_pred.shape
#    print Counter(y_pred)
    return np.mean(y_true == y_pred)

#print "Support vector machines:"
#print "%.3f" % accuracy(y, run_cv(X,y,SVC))
#print "Random forest:"
#print "%.3f" % accuracy(y, run_cv(X,y,RF))
print "K-nearest-neighbors:"
#print "%.3f" % accuracy(y, run_cv(X,y,KNN))

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y,run_cv(X,y,SVC))

print (cm[1,0]/cm[1].sum()+cm[0,0]/cm[0].sum())/2
y = np.array(y)
class_names = np.unique(y)
print np.bincount(y)
#confusion_matrices = [
#    ( "Support Vector Machines", confusion_matrix(y,run_cv(X,y,SVC)) ),
#    ( "Random Forest", confusion_matrix(y,run_cv(X,y,RF)) ),
#    ( "K-Nearest-Neighbors", confusion_matrix(y,run_cv(X,y,KNN)) ),
#]

# Pyplot code not included to reduce clutter
#from churn_display import draw_confusion_matrices
#from pandas_confusion import ConfusionMatrix
#cm = ConfusionMatrix(y,run_cv(X,y,SVC))
#print cm
#import matplotlib.pyplot as plt
#cm.plot()
#
#plt.show()
#draw_confusion_matrices(confusion_matrices,class_names)