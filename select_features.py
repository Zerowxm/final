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
def loadFile(file,header=0):
    df=pd.read_table(file,header=header)
    return df
    
def cat(obj):
   tmp=obj.copy()
   for col in tmp.columns:
         tmp[col]=tmp[col].astype('category')    
   return tmp
   
def test(df):
    df=df.fillna(0)
    o=df.select_dtypes(include=['object'])
    category=o.columns.values.tolist()
    o= o.apply(u.convert,axis='columns')
    df[category]=o
    u.treeClassifer(df,'churn')
    
def loadData():
    # comment start
    # read data
    #comment end
    data="orange_small_train.data"
    appetency='orange_small_train_appetency.labels'
    churn='orange_small_train_churn.labels'
    upselling='orange_small_train_upselling.labels'
    df=loadFile(data)
    appetency=loadFile(appetency,None)
    churn=loadFile(churn,None)
    upselling=loadFile(upselling,None)
    churn.columns=['churn']
    appetency.columns=['appetency']
    upselling.columns=['upselling']
    df=pd.concat([df,churn,appetency,upselling],axis=1) # concat data and label
    df=df.dropna(axis='columns',how='all')
    return df
#def handleTrain():
    
if __name__ == "__main__":
    df=loadData()
    labels=['churn','appetency','upselling']
    label='churn'
    temp=df.drop(df.dropna(axis='columns',how='any').columns.values,axis='columns').columns.values.tolist()
    null=pd.isnull(df[temp]).as_matrix().astype(np.int)
    missing_count=df.notnull().sum(axis=1)
    temp=[s + '_' for s in temp]
    null=pd.DataFrame(null,columns=temp)
    df=pd.concat([df,null],axis='columns')
#    df['missing_count']=missing_count
    # comment start
    # divide the data into two classes
    #comment end
    
    df_all=df.copy()
    df_all_cat=cat(df_all).describe()
    df_all_cat=df_all_cat.transpose()
    df_all_cat=df_all_cat[df_all_cat.unique>1]
    
    
    df=df[df_all_cat.index]
    df_all=df_all[df_all_cat.index]
    
    X_train, X_test, y_train, y_test=u.split(df_all,'churn')
    
    

    features=df_all.drop(labels,axis='columns').columns
    df_=df[df[label]==-1] # choose the label == -1
    df=df[df[label]==1]# choose the label == 1
    df_cat=cat(df).describe()
    df_cat=df_cat.transpose()
    df_n_cat=cat(df_).describe()
    df_n_cat=df_n_cat.transpose()
#    test(df_all)
       
    # comment start
    # sort the features by missing values'count
    #comment end
    num=df.isnull().sum().sort_values(ascending=True,kind='quicksort') # sort the columns by the missing values' count
    num_n=df_.isnull().sum().sort_values(ascending=True,kind='quicksort')
    num_per=df.isnull().sum(axis=1).sort_values(ascending=True,kind='quicksort') # every sample's  missing count
    num_per_des=num_per.describe()
    num_per_n=df_.isnull().sum(axis=1).sort_values(ascending=True,kind='quicksort') # every sample's  missing count
    num_per_n_des=num_per_n.describe()
    df_=df_.reindex(num_per_n.index)
#    num_per_n=num_per_n[num_per_n<num_per.min()]
    flag_all=True
    
    if (flag_all):
        dfd_n=df_
        dfd=df
    # comment start
    # extract the string features
    #comment end
    obj_all=df_all.select_dtypes(include=['object'])
    obj=dfd.select_dtypes(include=['object'])
    obj_n=dfd_n.select_dtypes(include=['object'])
    df_obj_cat=cat(obj).describe()
    df_obj_cat=df_obj_cat.transpose()
    
    df_obj_n_cat=cat(obj_n).describe()
    df_obj_n_cat=df_obj_n_cat.transpose()
    
    df_all_obj_cat=cat(obj_all).describe()
    df_all_obj_cat=df_all_obj_cat.transpose()
    category=obj.columns.values.tolist()  # string features' names
#    obj_all=obj_all.fillna('0')
#    obj_fill=obj_all.apply(u.convert_test,axis='index')
    numerical_all=df_all.drop(labels,axis='columns').select_dtypes(exclude=['object'])
#    df_all[category]=obj_fill
        
    dfd=dfd.drop(labels,axis='columns')
    dfd_n=dfd_n.drop(labels,axis='columns')
    numerical=dfd.select_dtypes(exclude=['object'])
    numerical_n=dfd_n.select_dtypes(exclude=['object'])
    numerical_category=numerical.columns.values.tolist()
    
    mean=numerical_all.mean()
    median=numerical_all.median()
    numerical_fill=numerical_all.fillna(mean,axis='rows')
#    df_all[numerical_category]=numerical_fill
    
    is_selected=True
    if(is_selected):
        arr=[]
        for col in category:
            unique=np.array(obj[col].unique())
            unique_n=np.array(obj_n[col].unique())
            unique_inter= u.intersection_count(unique,unique_n)
            arr.append(unique_inter)
        arr=np.array(arr)
        
        indices=np.where(arr<0.6)[0]
        if len(indices)!=0:
            features_categorical=[category[i] for i in indices ] 
#    u.selectF(df_all,'churn')
    select=False
    if(select):
        f1=[]
        for col in numerical_category:
            col_label=np.append(col,labels)
            df_col=df_all[col_label]
            f1.append(u.selectF(df_col,'churn'))
        np.savetxt('f1_all.txt',f1)
    # comment start
    # f1!!!!!!
    #comment end
    acc_f1=False
    if(acc_f1):
        f1=[]
        shape=[]
        positive=[]
        for col in category:
            col_label=np.append(col,labels)
            df_col=df_all[col_label]
            df_col=df_col.dropna(axis='rows') 
            df_col[col]=u.convert(df_col[col])
            shape.append( df_col.shape[0])
            f1.append(u.selectF(df_col,'churn'))
            positive.append(df_col[df_col[label]==1].shape[0])
        np.savetxt('f1_obj.txt',f1)
        np.savetxt('shape_obj.txt',shape)
        np.savetxt('positive_obj.txt',positive)
        
    acc_f1=False
    if(acc_f1):
        f1=[]
        shape=[]
        positive=[]
        for col in numerical_category:
            col_label=np.append(col,labels)
            df_col=df_all[col_label]
            df_col=df_col.dropna(axis='rows') 
            shape.append( df_col.shape[0])
            f1.append(u.selectF(df_col,'churn'))
            positive.append(df_col[df_col[label]==1].shape[0])
        np.savetxt('f1.txt',f1)
        np.savetxt('shape.txt',shape)
        np.savetxt('positive.txt',positive)
    
    positive=[]
    to_acc=False
    if(to_acc):
        for col in numerical_category:
            col=np.append(col,label)
            df_col=df_all[col]
            positive.append(df_col.dropna(axis='rows')[df_all[label]==1].shape[0])
        np.savetxt('positive.txt',positive)
    bug=False
    if(bug):
        positive=np.loadtxt('positive.txt')
        f1=np.loadtxt('f1.txt')
        shape=np.loadtxt('shape.txt')
        f1_df=pd.DataFrame(f1,columns=['f1','accuracy','precision','recall','negative_a','positive_a'])
        
        f1_df['count_ratio']=pd.Series(shape)/df_all.shape[0]
        f1_df['positive']=pd.Series(positive)
        f1_df['positive_ratio']=pd.Series(positive).divide(shape)
        f1_df=f1_df.transpose()
        f1_df.columns=numerical_category
        f1_df=f1_df.transpose()
    #    f1_df=f1_df[f1_df.positive_a!=0]
        f1_df_s=u.standardize_df(f1_df)
        
        positive=np.loadtxt('positive_obj.txt')
        f1=np.loadtxt('f1_obj.txt')
        shape=np.loadtxt('shape_obj.txt')
        f1_obj_df=pd.DataFrame(f1,columns=['f1','accuracy','precision','recall','negative_a','positive_a'])
        
        f1_obj_df['count_ratio']=pd.Series(shape)/df_all.shape[0]
        f1_obj_df['positive_count']=pd.Series(positive)
        f1_obj_df['positive_ratio']=pd.Series(positive).divide(shape)
        f1_obj_df=f1_obj_df.transpose()
        f1_obj_df.columns=category
        f1_obj_df=f1_obj_df.transpose()
        f1_obj_df=f1_obj_df[f1_obj_df.positive_a!=0]
        f1_obj_df_s=u.standardize_df(f1_obj_df)
    
#    dfd=pd.concat([dfd,classes],axis='columns')
#    
#    dfd_n=pd.concat([dfd_n,classes_n],axis='columns')
#    train=pd.concat([dfd,dfd_n])
#    train_features=train[category].apply(u.convert,axis='columns')
#    
#    train[category]=train_features
#    
#    category.extend(labels)
#    train=train[category]
