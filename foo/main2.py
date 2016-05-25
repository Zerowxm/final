# -*- coding: utf-8 -*-
"""
Created on Wed May 11 16:21:54 2016

@author: Zero
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 05 16:18:21 2016

@author: Zero
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils as u
def loadData(file,header=0):
    df=pd.read_table(file,header=header)
    return df
    
def cat(obj):
   for col in obj.columns:
         obj[col]=obj[col].astype('category')    
   return obj
   
def plot_rfe(X,label):
    y=X[label]
    X=X.drop(['churn','appetency','upselling',label],axis='columns')
    from sklearn.svm import SVC
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.feature_selection import RFECV
    # Build a classification task using 3 informative features
#    X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
#                               n_redundant=2, n_repeated=0, n_classes=8,
#                               n_clusters_per_class=1, random_state=0)
    
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, 2),
                  scoring='accuracy')
    rfecv.fit(X, y)
    
    print("Optimal number of features : %d" % rfecv.n_features_)
    
    # Plot number of features VS. cross-val5idation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
   
if __name__ == "__main__":
    data="orange_small_train.data"
    appetency='orange_small_train_appetency.labels'
    churn='orange_small_train_churn.labels'
    upselling='orange_small_train_upselling.labels'
    df=loadData(data)
    appetency=loadData(appetency,None)
    churn=loadData(churn,None)
    upselling=loadData(upselling,None)
    churn.columns=['churn']
    appetency.columns=['appetency']
    upselling.columns=['upselling']
    labels=pd.concat([churn,appetency,upselling],axis=1)
    labels_des=labels.describe()
    labels_des.drop('count',axis='rows').plot()
    labels_cat=cat(labels)
    cat_des=labels_cat.describe()
    df=pd.concat([df,labels],axis=1)
    df_churn=df[df.churn==1]
    
    
    