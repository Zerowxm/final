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
import stacked_generalization
#df=handle.handle_missing(df_all,label,is_common=True)
#y=df[label],X=df.drop(labels,axis='columns')
#X_train, X_test, y_train, y_test=u.split(X,y)
X_trai=X_train.copy().as_matrix()
y_trai=y_train.copy().as_matrix()
X_tes=X_test.copy()
columns=X_train.columns.values
oversample=False
from sklearn.grid_search import ParameterGrid
grid = ParameterGrid({"k": [3,4,5,6,7],
                          "ratio": [0.1,0.2,0.3,0.4,0.5] })
def plot_scores(scores):
    scores=pd.DataFrame(scores,index=['RT','gbc','ets','lgr','adboost','dt','voting'],columns=['auc','accuracy','f1','precision','recall','kappa'])
    scores.plot()
if(oversample):
#    scores=[]
#    for params in grid:
#        print params
        adsn = ADASYN(imb_threshold=0.8,ratio=1)
        X_trai, y_trai = adsn.fit_transform(X_trai,y_trai)  # your imbalanced dataset is in X,y
#        X_trai,y_trai=u.test_rest(X_trai,y_trai) 
        u.all_lassifer(X_trai,y_trai,X_test,y_test)
#        u.all_lassifer(X_trai,y_trai,X_tes,y_tes)
#        scores.append(u.boostingClassifier(X_trai,y_trai,X_tes,y_tes))
#        scroes=pd.DataFrame(scores,columns=['auc','f1','accuracy','precision','recall','kappa'])
#        print Counter(y_trai)
else:
#    scores=[]
    predcit=[]
    grid = ParameterGrid({'c':[0] })
    for params in grid:
        print params
        X_trai,y_trai=u.test_rest(X_trai,y_trai,ratio=3,**params)
        X_trai,y_trai=u.test_smote(X_trai,y_trai,c=0)
        
        select=1
        if(select==1):
            X_trai=pd.DataFrame(X_trai,columns=columns)
            columns=u.gbc_features(X_trai,y_trai,columns)
            
#        X_trai,y_trai=u.test_smote(X_trai.as_matrix(),y_trai,0)
        elif(select==2):
            columns=features_to_select[features_to_select.iloc[:]!=0].index.values.tolist()
        if(select):
            X_trai=pd.DataFrame(X_trai,columns=columns)
            X_trai=X_trai[columns]
#            scores=u.gbClassifier(X_trai,y_trai,X_test[columns],y_test) 
            u.abClassifier(X_trai,y_trai,X_test[columns],y_test) 
        else:   
#            u.selectFeatures_train(X_trai,y_trai,k=20,p=60)
#            X_trai=u.selectFeatures(X_trai,t=1)
#            X_tes=u.selectFeatures(X_test,t=1)
#            X_trai,X_tes=u.standardize(X_trai,X_tes)
#            u.test(X_trai,y_trai,X_tes,y_test)
#            X_trai,y_trai=u.test_rest(X_trai,y_trai,c=9)
            best_score = 0.0
            
            # run many times to get a better result, it's not quite stable.
#            for i in xrange(1):
#                print 'Iteration [%s]' % (i)
#                score = stacked_generalization.run(X_trai,y_trai,X_tes,y_test)
#                best_score = max(best_score, score)
#                print
#                
#            print 'Best score = %s' % (best_score)
            score=u.gbClassifier(X_trai,y_trai,X_test,y_test,0)
#            u.cross_val(X_trai.as_matrix(),y_trai)
#            X_trai,X_tes=u.stackingClassifier(X_trai,y_trai,X_tes,y_test) 
#            scores=u.test(X_trai,y_trai,X_tes,y_test) 
#    adsn = ADASYN(imb_threshold=0.8,ratio=2)
#    X_trai, y_trai = adsn.fit_transform(X_trai,y_trai)
#            for i in range(X_trai.shape[0]):
#                X=X_trai[i] 
#                y=y_trai[i]
#                X,y=u.test_rest(X,y,c=9)
#                u.gbClassifier(X,y,X_test,y_test)
#        X_trai,X_tes=u.stackingClassifier(X_trai,y_trai,X_test[columns],y_test)
#        score=u.gbClassifier(X_trai,y_trai,X_test,y_test)
#        u.all_lassifer(X_trai,y_trai,X_tes,y_test)
#        scores.append(score)
#        predcit.append(predcit_)
#    u.gbc(X_trai.as_matrix(),y_trai.as_matrix(),columns)
print Counter(y_trai)
#u.treeClassifer(X,y)

#from collections import Counter

# In many applications you may want to keep artificial data separately
# adsn.index_new is a list that holds the indexes of these examples
