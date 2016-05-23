# -*- coding: utf-8 -*-
"""
Created on Tue May 17 23:07:39 2016

@author: Zero
"""
import pandas as pd
from adasyn import ADASYN
import utils as u
from collections import Counter
import handle
#df=handle.handle_missing(df_all,label,is_common=True)
#y=df[label],X=df.drop(labels,axis='columns')
#X_train, X_test, y_train, y_test=u.split(X,y)
X_trai=X_train.copy().as_matrix()
y_trai=y_train.copy().as_matrix()
columns=X_train.columns.values
oversample=False
from sklearn.grid_search import ParameterGrid
grid = ParameterGrid({"k": [3,4,5,6,7],
                          "ratio": [0.1,0.2,0.3,0.4,0.5] })
   
if(oversample):
#    scores=[]
#    for params in grid:
#        print params
        adsn = ADASYN(imb_threshold=0.8,ratio=4)
        X_trai, y_trai = adsn.fit_transform(X_trai,y_trai)  # your imbalanced dataset is in X,y
#        X_trai,y_trai=u.test_rest(X_trai,y_trai) 
        u.all_lassifer(X_trai,y_trai,X_test,y_test)
#        u.all_lassifer(X_trai,y_trai,X_tes,y_tes)
#        scores.append(u.boostingClassifier(X_trai,y_trai,X_tes,y_tes))
#        scroes=pd.DataFrame(scores,columns=['auc','f1','accuracy','precision','recall','kappa'])
#        print Counter(y_trai)
else:
    scores=[]
    predcit=[]
    grid = ParameterGrid({'c':[0] })
    for params in grid:
        print params
        X_trai,y_trai=u.test_rest(X_trai,y_trai,ratio=6,**params) 
        X_trai,y_trai=u.test_smote(X_trai,y_trai,0)
        X_trai=pd.DataFrame(X_trai,columns=columns)
#        columns=u.gbc(X_trai,y_trai,columns)
#        X_trai=X_trai[columns]
    
#    adsn = ADASYN(imb_threshold=0.8,ratio=2)
#    X_trai, y_trai = adsn.fit_transform(X_trai,y_trai)
#        for i in range(9):
#            scores.append(u.gbClassifier(X_trai[i],y_trai[i],X_tes,y_tes))
        X_trai,X_tes=u.stackingClassifier(X_trai,y_trai,X_test[columns],y_test)
#        predcit_,score=u.gbClassifier(X_trai,y_trai,X_tes,y_test)
        u.all_lassifer(X_trai,y_trai,X_tes,y_test)
#        scores.append(score)
#        predcit.append(predcit_)
#    u.gbc(X_trai.as_matrix(),y_trai.as_matrix(),columns)
print Counter(y_trai)
#u.treeClassifer(X,y)

#from collections import Counter

# In many applications you may want to keep artificial data separately
# adsn.index_new is a list that holds the indexes of these examples
