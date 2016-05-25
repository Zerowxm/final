# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:16:30 2016

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
    
    # Plot number of features VS. cross-validation scores
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
    
    df=pd.concat([df,churn,appetency,upselling],axis=1)
    
    df=df.dropna(axis='columns',how='all')
    num=df.isnull().sum().sort_values(ascending=True,kind='quicksort')
    features=np.array(num.index.tolist())
    df=df[features]
    
    dfd_any=df.dropna(axis='columns',how='any')
    dfd_n=df.drop(dfd_any.columns,axis='columns')
    label=dfd_n[dfd_n.columns[0]]
    obj_any=dfd_any.select_dtypes(include=['object'])
    catagory_any=obj_any.columns.values
    numerical=dfd_any.select_dtypes(exclude=['object'])
    num_des=numerical.describe()
    
#    print obj.isnull().sum().sum()
    obj_any.loc[:,catagory_any]= obj_any[catagory_any].apply(u.convert,axis='columns')
    obj=cat(obj_any)
#    obj=obj.apply(u.inpute)
#    dfd.fillna(method='bfill')
    
    
#    print obj.isnull().sum().sum()
    dfd_any.loc[:,catagory_any]=dfd_any[catagory_any].apply(u.convert,axis='columns')
    dfd_any=cat(dfd_any)
#    u.treeClassifer(dfd_any,'upselling')
#    plot_rfe(dfd_any,'upselling')
    cat = obj.describe()
    cat1=dfd_any.describe()
    data=pd.concat([dfd_any,label],axis=1)
    train=data[pd.notnull(data[label.name])]
    test=data[pd.isnull(data[label.name])]
    label=label[pd.notnull(label)]
#    u.treeClassifer(data,label.name)
    print 'begin'
#    u.predict(test,label.name)
#    label_c= label.astype('category').describe()
    
