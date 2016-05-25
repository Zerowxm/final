# -*- coding: utf-8 -*-
"""
Created on Mon May 23 22:30:31 2016

@author: Zero
"""

import pandas as pd
import numpy as np
import utils as u
#u.plot_corr(X_train.iloc[:,-28:])
#scores=np.loadtxt('scores.txt')
features_to_select=pd.read_csv('scores.txt',delimiter=r"\s+",index_col=0)
u.standardize_df(features_to_select).boxplot()

#f1=np.loadtxt('f1.txt')
#features=df_all.drop(labels,axis='columns').columns.values
#f1_df=pd.DataFrame(f1,columns=['f1','accuracy','precision','recall','negative_a','positive_a'])
#s=pd.DataFrame(scores,columns=['auc','f1','accuracy','precision','recall','kappa'])
#s.index=['rt','gbc','ets','lr','ab','dt','voting']
#s.plot()
#s.to_csv("scores.csv")
#scores.plot()
#,delim_whitespace=True,