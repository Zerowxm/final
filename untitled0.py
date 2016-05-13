# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:53:47 2016

@author: Zero
"""
from sklearn.externals import joblib

rfecv=joblib.load( 'rfecv.pkl.pkl') 
su=rfecv.support_