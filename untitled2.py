# -*- coding: utf-8 -*-
"""
Created on Fri May 20 19:59:22 2016

@author: Zero
"""
import matplotlib.pyplot as plt
import pandas as pd 
import utils 
test=pd.read_csv('test.csv',header=None)
#y_=pd.DataFrame(y_test.copy())
##y_=pd.concat([y_,test],axis='columns')
#y_['predict']=test.values
#y_=y_[y_.churn==1]
#test=X_test.loc[y_.index]
#ytest=pd.concat([test['missing_count'],y_],axis='columns')
#ytest=utils.standardize_df(ytest)
#ytest.iloc[:20].plot()
#test['label']=y_test.values
