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
def handle_missing(df_all,label):
    labels=['churn','appetency','upselling']
    df_fill=df_all.copy()
    df_all_cat=u.to_category(df_all).describe()
    df_all_cat=df_all_cat.transpose()
    df_all_cat=df_all_cat[df_all_cat.unique>1]
    df_all=df_all[df_all_cat.index]
    df_=df_all[df_all[label]==-1] # choose the label == -1
    df=df_all[df_all[label]==1]# choose the label == 1
    
    category=df_all.select_dtypes(include=['object']).columns.values.tolist()  
    
    df_cat=u.to_category(df).describe()
    df_cat=df_cat.transpose()
    df_n_cat=u.to_category(df_).describe()
    df_n_cat=df_n_cat.transpose()
    
    
    df_all_cat['freq_ratio']=df_all_cat['freq']/df_all_cat['count']
    df_all_cat['unique_ratio']=df_all_cat['unique']/df_all_cat['count']
    features_fill_freq=df_all_cat[df_all_cat.freq_ratio>=0.5].drop(labels,axis='rows').index.values.tolist()
    features_fill_median=df_all_cat.drop(category,axis='rows')[(df_all_cat.freq_ratio<0.5) & (df_all_cat.unique_ratio>=0.5)].index.values.tolist()
    features_fill_mean=df_all_cat.drop(category,axis='rows')[(df_all_cat.freq_ratio<0.5) & (df_all_cat.unique_ratio<0.5)].index.values.tolist()
    median= df_all[features_fill_median].median()
    mean= df_all[features_fill_mean].mean()
    mean_= df[features_fill_mean].mean()
    mean_n= df_[features_fill_mean].mean()
    mean=(mean_+mean_n)/2
    #mean_=pd.DataFrame([mean,mean_,mean_n]).transpose().mean(axis='columns')
    freq=df_all_cat[df_all_cat.freq_ratio>=0.5].drop(labels,axis='rows')['top']
    df_fill[features_fill_freq]=df_all[features_fill_freq].fillna(freq)
    df_fill[features_fill_median]=df_all[features_fill_median].fillna(median)
    df_fill[features_fill_mean]=df_all[features_fill_mean].fillna(mean)
    df_fill=df_fill.fillna('0')
    return df_fill
df_fill=handle_missing(df_all,label)
X_train, X_test, y_train, y_test=u.split(df_fill,'churn')
#df_fill=df_all.copy()
#df_all_cat['freq_ratio']=df_all_cat['freq']/df_all_cat['count']
#df_all_cat['unique_ratio']=df_all_cat['unique']/df_all_cat['count']
#features_fill_freq=df_all_cat[df_all_cat.freq_ratio>=0.5].drop(labels,axis='rows').index.values.tolist()
#features_fill_median=df_all_cat.drop(category,axis='rows')[(df_all_cat.freq_ratio<0.5) & (df_all_cat.unique_ratio>=0.5)].index.values.tolist()
#features_fill_mean=df_all_cat.drop(category,axis='rows')[(df_all_cat.freq_ratio<0.5) & (df_all_cat.unique_ratio<0.5)].index.values.tolist()
#median= df_all[features_fill_median].median()
#mean= df_all[features_fill_mean].mean()
#mean_= df[features_fill_mean].mean()
#mean_n= df_[features_fill_mean].mean()
##mean_=pd.DataFrame([mean,mean_,mean_n]).transpose().mean(axis='columns')
#freq=df_all_cat[df_all_cat.freq_ratio>=0.5].drop(labels,axis='rows')['top']
#df_fill[features_fill_freq]=df_all[features_fill_freq].fillna(freq)
#df_fill[features_fill_median]=df_all[features_fill_median].fillna(median)
#df_fill[features_fill_mean]=df_all[features_fill_mean].fillna(mean)
#df_fill=df_fill.fillna(0)


df_obj_cat['freq_ratio']=df_obj_cat['freq']/df_obj_cat['count']
df_obj_cat['unique_ratio']=df_obj_cat['unique']/df_obj_cat['count']
df_obj_cat=df_obj_cat[df_obj_cat.freq_ratio>=0.5]

df_obj_n_cat['freq_ratio']=df_obj_n_cat['freq']/df_obj_n_cat['count']
#df_obj_n_cat=df_obj_n_cat[df_obj_n_cat.freq_ratio>0.5]

df_all_obj_cat['freq_ratio']=df_all_obj_cat['freq']/df_all_obj_cat['count']
df_all_obj_cat['unique_ratio']=df_all_obj_cat['unique']/df_all_obj_cat['count']

#df_all_obj_cat=df_all_obj_cat[df_obj_cat.freq_ratio>0.5]

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