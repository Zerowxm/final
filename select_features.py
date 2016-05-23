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
    label='upselling'
    df=handle.loadData()
    df_all=df
    df_fill=handle.handle_missing(df,label,is_common=True)
    y=df_fill[label]
    X=df_fill.drop(labels,axis='columns')
#    X=X[u.gbc(X,y,X.columns)]
    X_train, X_test, y_train, y_test=u.split(X,y)
#    
    # comment start
    # divide the data into two classes
    #comment end
    
    