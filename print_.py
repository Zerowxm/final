# -*- coding: utf-8 -*-
"""
Created on Tue May 17 23:07:39 2016

@author: Zero
"""
import pandas as pd
from adasyn import ADASYN
import utils as u
from collections import Counter
#y=train[label]
#X_train, X_test, y_train, y_test=u.split(df_all,'churn')
X_trai=X_train.copy()
y_trai=y_train.copy()
#X=train.drop(['churn','appetency','upselling',label],axis='columns')
X_trai[category]=X_trai[category].fillna('0')
X_trai[category]=u.normalize_df(X_trai[category].apply(u.convert_test,axis='index'))
mean=X_trai.mean()
median=X_trai.median()
X_trai=X_trai.fillna(mean,axis='rows')


X_tes=X_test.copy()
y_tes=y_test.copy()
X_tes[category]=X_tes[category].fillna('0')
X_tes[category]=u.normalize_df(X_tes[category].apply(u.convert_test,axis='index'))
X_tes=X_tes.fillna(mean,axis='rows')

select=False
if(select):
    X_trai=X_trai[features_selected]
    X_tes=X_tes[features_selected]

oversample=False
from sklearn.grid_search import ParameterGrid
grid = ParameterGrid({"k": [3,4,5,6,7],
                          "ratio": [0.1,0.2,0.3,0.4,0.5] })
columns=X_trai.columns.values   
#X_trai,y_trai=u.test_rest(X_trai.as_matrix(),y_trai.as_matrix()) 
#y_trai=x_smote['churn']
#X_trai=x_smote.drop('churn',axis='columns')        
if(oversample):
    scores=[]
    for params in grid:
        print params
        adsn = ADASYN(imb_threshold=0.8,**params)
        X_trai, y_trai = adsn.fit_transform(X_trai,y_trai)  # your imbalanced dataset is in X,y
        X_trai,y_trai=u.test_rest(X_trai,y_trai) 
        scores.append(u.boostingClassifier(X_trai,y_trai,X_tes,y_tes))
        scroes=pd.DataFrame(scores,columns=['auc','f1','accuracy','precision','recall','kappa'])
        print Counter(y_trai)
else:
    X_trai,y_trai=u.test_rest(X_trai.as_matrix(),y_trai.as_matrix(),2) 
    u.boostingClassifier(X_trai,y_trai,X_tes,y_tes)
print Counter(y_trai)
#u.treeClassifer(new,'churn')

#from collections import Counter

# In many applications you may want to keep artificial data separately
# adsn.index_new is a list that holds the indexes of these examples
