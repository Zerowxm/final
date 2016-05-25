# -*- coding: utf-8 -*-
"""
Created on Wed May 18 21:02:35 2016

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
from sklearn import tree
from sklearn.externals import joblib
import handle
    
if __name__ == "__main__":
    labels=['churn','appetency','upselling']
    label='appetency'
    df=handle.loadData()
    df_all=df
    df_fill=handle.handle_missing(df,label,is_common=True,replace=0)
    y_=df_fill[label]
    X_=df_fill.drop(labels,axis='columns')
#    X=X[u.gbc(X,y,X.columns)]
    X_train, X_test, y_train, y_test=u.split(X_,y_,test_size=0.1)
    df_split=handle.split_data(pd.concat([X_train, y_train],axis='columns'),label)
    
    # comment start
    # divide the data into two classes
    #comment end
    
    