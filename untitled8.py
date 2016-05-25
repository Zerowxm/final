# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:27:24 2016

@author: Zero
"""
import utils as u
import numpy  as np
if __name__ == '__main__':
    test=pd.read_csv('y_submission.csv',header=None)
#    print u.roc_auc_score(y_test,test,average='micro', sample_weight=None)
    print u.roc_auc_score(y_test,test,average='macro', sample_weight=None)
#    print u.roc_auc_score(y_test,y_submission,average='weighted', sample_weight=None)
#    print u.roc_auc_score(y_test,y_submission,average='samples', sample_weight=None)
#    for i in len(df_split):
#        X_train=df_split[i].drop(label,axis='columns')
#        X_test=df_split[i][label]