# -*- coding: utf-8 -*-
"""
Created on Mon May 23 21:15:47 2016

@author: Zero
"""

scores=pd.DataFrame(scores,index=['RT','gbc','ets','lgr','adboost','dt','voting'],columns=['auc','accuracy','f1','precision','recall','kappa'])
scores.plot()