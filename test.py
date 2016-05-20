# -*- coding: utf-8 -*-
"""
Created on Sat May 14 11:06:38 2016

@author: zero
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
def loadData(file,header=0):
    df=pd.read_table(file,header=header)
    return df
    
def cat(obj):
   for col in obj.columns:
         obj[col]=obj[col].astype('category')    
   return obj
   
from sklearn import preprocessing
def convert(x):
     le = preprocessing.LabelEncoder()
     return le.fit_transform(x)   
if __name__ == "__main__":
    # comment start
    # read data
    #comment end
    data="orange_small_train.data"
    appetency='orange_small_train_appetency.labels'
    churn='orange_small_train_churn.labels'
    upselling='orange_small_train_upselling.labels'
    df=loadData(data)
    appetency=loadData(appetency,None)
    churn=loadData(churn,None)
    upselling=loadData(upselling,None)
    churn.columns=['churn']
    appetency.columns=['appetency']
    upselling.columns=['upselling']
    df=pd.concat([df,churn],axis=1) # concat data and label
    df=df.dropna(axis='columns',how='all')
    obj=df.select_dtypes(include=['object'])
    obj_= obj.apply(convert,axis='columns')
#    obj_=obj_.replace(0,'?')
#    obj = obj.fillna('?')
    features=obj_.columns.values.tolist()
    values=obj_.values.tolist()[:2]
    obj_=pd.concat([obj_,churn],axis=1)
    mat=obj_.as_matrix()
#    mat[mat==0]=np.NaN
    print type(mat)
#    mat=np.asarray(values)
    import ut
    import Orange
    from Orange.data import Domain
    f=[Orange.feature.Discrete('%s'%x) for x in features]
    c=Orange.feature.Discrete("churn")
    domain=Orange.data.Domain(f+[c])
    a = np.array([[1, '?', 3, 4, 5], [5, 4, 3, 2, 1]])
    loe = [["d", "1", "1", "1", "?", "1",'?', "1"],
       ["3", "1", "1", "2","1", "1", '?',"0"],
       ["3", "3", "1", "2", "2", "1",2 ,"1"]
      ]
#    data=ut.df2table(obj)
    data=Orange.data.Table(domain,mat)
    print data.domain.features
    for x in data.domain.features:
        n_miss = sum(1 for d in data if d[x].is_special())
        print "%4.1f%% %s" % (100.*n_miss/len(data), x.name)
#    obj=ut.table2df(data)
#    domain = Domain(features)
    
#    d = Orange.data.Domain([Orange.feature.String('a')]+[Orange.feature.Continuous('a%i' % x) for x in range(5)])
#    d_f=d.features
#    d_l=d.class_var
    
#    t=Orange.data.Table(d,)