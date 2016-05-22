# -*- coding: utf-8 -*-
"""
Created on Sat May 21 01:00:09 2016

@author: Zero
"""
import utils as u
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('fivethirtyeight') # Good looking plots
pd.set_option('display.max_columns', None) # Display any number of columns
#bins = np.linspace(df.Var1.min(), df.Var1.max(), 10)
#groups = pd.cut(df.Var1, bins)
#print groups
df_all_cat['ratio']=df_all_cat['freq']/df_all_cat['count']
#df_all_=df_all_cat[df_all_cat.ratio>0.5]

df_obj_cat['ratio']=df_obj_cat['freq']/df_obj_cat['count']
#df_obj_cat=df_obj_cat[df_obj_cat.ratio>0.5]

df_obj_n_cat['ratio']=df_obj_n_cat['freq']/df_obj_n_cat['count']
#df_obj_n_cat=df_obj_n_cat[df_obj_n_cat.ratio>0.5]

df_all_obj_cat['ratio']=df_all_obj_cat['freq']/df_all_obj_cat['count']
#df_all_obj_cat=df_all_obj_cat[df_obj_cat.ratio>0.5]

#df_cat=cat(df).describe()
#df_cat=df_cat.transpose()
#df_n_cat=cat(df_).describe()
#df_n_cat=df_n_cat.transpose()
df_var=pd.read_csv('var.txt',delimiter=' ',header=None)
print u.intersection(df_all_.index.values,df_var[4])
#num_all=df_all.isnull().sum(axis=1).sort_values(ascending=True,kind='quicksort')
#D=Counter(num_all)
#D_=Counter(num_per)
#D_n=Counter(num_per_n)
#plt.bar(range(len(D)), D.values(), align='center')
#plt.xticks(range(len(D)), D.keys())
#fig, ax = plt.subplots()
#ax.bar(D_.keys(),u.preprocessing.scale(D_.values()), 0.8, color='r')
#ax.bar(D_n.keys(),u.preprocessing.scale(D_n.values()), 0.5, color='g')
#plt.xlabel('missing count')
#plt.ylabel('sample count')
#ax.legend(('1','-1'))

#u.standardize_df(df_all_cat.drop('top',axis='columns')).plot()
#plt.show()
#num_all_s=u.standardize_df(pd.DataFrame(num_all,columns=['missing count']))
#num_all.plot()
#num_all_s.plot()
#temp_s[['var','var_n']].plot()
#temp_s[['mean','mean_n']].plot()
#temp_s.plot.scatter(x='var', y='var_n',c='c', s=50);
#temp_s[['min','min_n','max','max_n']].plot()
#temp_s.drop(['count_','count_n','mean','mean_n','min','min_n','max','max_n'],axis='columns').plot()
#if __name__ == "__main__":
#    numerical_cate=cat(numerical).describe().transpose()
#    numerical_n_cate=cat(numerical_n).describe().transpose()
#    numerical_cate['type']=pd.Series(numerical.dtypes)
#    numerical_n_cate['type']=pd.Series(numerical_n.dtypes)
#    numerical_des=numerical.describe()
#    numerical_n_des=numerical_n.describe()
#    bug=True
#    if(bug):
#        count=numerical_cate['count']/numerical.shape[0]
#        count.name='count_'
#        count_n=numerical_n_cate['count']/numerical_n.shape[0]
#        count_n.name='count_n'
#        var=numerical.var()
#        var.name='var'
#        std=numerical.std()
#        std.name='std'
#        mean=numerical.mean()
#        mean.name='mean'
#        mean_n=numerical_n.mean()
#        mean_n.name='mean_n'
#        var_n=numerical_n.var()  
#        var_n.name='var_n'
#        std_n=numerical_n.std()
#        std_n.name='std_n'
#        max_n=numerical_n.max()
#        max_n.name='max_n'
#        max_=numerical.max()
#        max_.name='max'
#        min_n=numerical_n.min()
#        min_n.name='min_n'
#        min_=numerical.min()
#        min_.name='min'
#        temp=pd.DataFrame([count,count_n,var,var_n,std,std_n,mean,mean_n,min_,min_n,max_,max_n]).transpose()
#        temp_s=u.standardize_df(temp)