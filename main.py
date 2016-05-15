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
   tmp=obj.copy()
   for col in tmp.columns:
         tmp[col]=tmp[col].astype('category')    
   return tmp
   

   
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
    df=pd.concat([df,churn,appetency,upselling],axis=1) # concat data and label
    # comment start
    # divide the data into two classes
    #comment end
    df=df.dropna(axis='columns',how='all')
    df_all=df.copy()
    df_des=df.describe()
    df_rank=df.rank() # rank the columns
    df_=df[df.churn==-1] # choose the label == -1
    df=df[df.churn==1]# choose the label == 1
   
    null=pd.isnull(df) 
    isnull=df.isnull().any()
    
    
    # comment start
    # sort the features by missing values'count
    #comment end
    num=df.isnull().sum().sort_values(ascending=True,kind='quicksort') # sort the columns by the missing values' count
    num_n=df_.isnull().sum().sort_values(ascending=True,kind='quicksort')
#    thres=40000    
#    num_=num[num<thres] # select data by shres
    num_per=df.notnull().sum(axis=1).sort_values(ascending=True,kind='quicksort') # every sample's  missing count
    num_per_des=num_per.describe()
    num_per_n=df_.notnull().sum(axis=1).sort_values(ascending=True,kind='quicksort') # every sample's  missing count
    num_per_n_des=num_per_n.describe()
    num_per_n=num_per_n[num_per_n<num_per.min()].index
    min_sam=df_.loc[num_per_n]
    mean_f=num_per.mean() 
    minn= num_per[num_per>=num_per.mean()]
    features=np.array(num.index.tolist())
    f=['Var126','Var29','Var130','Var201','Var90','Var192','Var138','Var113','Var74','Var13',
   'Var189','Var205','Var73','Var211','Var199','Var212','Var217','Var2','Var218','Var81',
   'churn','appetency','upselling']
   
    # comment start
    # reoeder the columns
    #comment end
    df=df[features] 
    df_=df_[features]
    df_tmp=df.copy()
    df_tmp1=df_.copy()
#    df=df.iloc[minn.index,:].dropna(axis='columns',how='any')
    
    # comment start
    # get dataframe describe
    #comment end
    des_p=df.describe()
    des_n=df_.describe()
    
    # comment start
    # choose the features by the missing values'shres , p = 1 n=-1
    #comment end
    a=0.8
    des_p_count=des_p.loc['count']/df.shape[0]
    des_p_count=des_p_count[des_p_count>a]
    des_n_count=des_n.loc['count']/df_.shape[0]
    des_n_count=des_n_count[des_n_count>a]
    des_diff= list(set(des_p_count.index.values).difference(
    set(des_n_count.index.values))) # diff the two classes features
#    des_p=des_p[des_diff]
    
    #[ 83 134 188 162 109  94 101 155 135 137] [175  94  56 134 211  66  74  73  72  71]
    #[ 4 12  9  8 10 16  1 18 19 17] [ 1  4 12 13 19  8  2  3  5  6]
    # comment start
    # choose the features
    #comment end
    flag_all=True
    labels=['churn','appetency','upselling']
#    fea=[83,134, 188, 162, 109,  94, 101, 155, 135 ,137,175,56,211,66]
    des_diff=np.append(des_diff,labels)
#    des_diff=['Var8','Var6','Var2','Var21','Var3','Var12','Var20','Var4','Var18','churn','appetency','upselling']
    if (flag_all):
        dfd_n=df_
        dfd=df
    else:
        dfd_n=df_[des_diff]
        dfd=df[des_diff]
#    cate_p=cat(dfd).describe()
#    cate_p.loc['count',:]=cate_p.loc['count']/cate_p.shape[0]
#    cate_n=cat(dfd_n).describe()
#    cate_n.loc['count':,]=cate_n.loc['count']/cate_n.shape[0]
    # comment start
    # preprocess the data
    #comment end

#    dfd_any=df.dropna(axis='columns',how='any')
    
    # comment start
    # extract the string features
    #comment end
    obj=dfd.select_dtypes(include=['object'])
    obj_n=dfd_n.select_dtypes(include=['object'])
    category=obj.columns.values.tolist()  # string features' names
    
    is_selected=True
    if(is_selected):
        arr=[]
        for col in category:
            unique=np.array(obj[col].unique())
            unique_n=np.array(obj_n[col].unique())
            unique_inter= u.intersection_count(unique,unique_n)
            arr.append(unique_inter)
        arr=np.array(arr)
        indices=np.where(arr>0.8)[0]
        category_del=[category[i] for i in indices ] 
        category=[x for x in category if x not in category_del]
        obj=obj.drop(category_del,axis='columns')
        obj_n=obj_n.drop(category_del,axis='columns')
        dfd=dfd.drop(category_del,axis='columns')
        dfd_n=dfd_n.drop(category_del,axis='columns')
    
    if(obj.shape[1]!=0):
        cate_p=cat(obj).describe()
        cate_p.loc['count',:]=cate_p.loc['count']/obj.shape[0]
        cate_n=cat(obj_n).describe()
        cate_n.loc['count',:]=cate_n.loc['count']/obj_n.shape[0]
    
    
    # comment start
    # extract the numerical feautures
    #comment end
    is_drop=False
    if(is_drop):
        classes=dfd[labels].reset_index(drop=True)
        classes_n=dfd_n[labels].reset_index(drop=True)
    else:
        classes=dfd[labels]
        classes_n=dfd_n[labels]
    dfd=dfd.drop(labels,axis='columns')
    dfd_n=dfd_n.drop(labels,axis='columns')
    numerical=dfd.select_dtypes(exclude=['object'])
    numerical_n=dfd_n.select_dtypes(exclude=['object'])
    numerical_cate=cat(numerical).describe()
    numerical_n_cate=cat(numerical_n).describe()
    numerical_des=numerical.describe()
    numerical_n_des=numerical_n.describe()
    numerical_unique=numerical.iloc[:,0].unique()
#    var=u.normalize_df(numerical).var()
#    var_n=u.normalize_df(numerical_n).var() 
    bug=False
    if(bug):
        count=numerical_cate.loc['count']/numerical.shape[0]
        count.name='count'
        count_n=numerical_n_cate.loc['count']/numerical_n.shape[0]
        count_n.name='count_n'
        var=numerical.var()
        var.name='var'
        std=numerical.std()
        std.name='std'
        mean=numerical.mean()
        mean.name='mean'
        mean_n=numerical_n.mean()
        mean_n.name='mean_n'
        var_n=numerical_n.var()  
        var_n.name='var_n'
        std_n=numerical_n.std()
        std_n.name='std_n'
        max_n=numerical_n.max()
        max_n.name='max_n'
        max_=numerical.max()
        max_.name='max'
        min_n=numerical_n.min()
        min_n.name='min_n'
        min_=numerical.min()
        min_.name='min'
        temp=pd.DataFrame([count,count_n,var,var_n,std,std_n,mean,mean_n,min_,min_n,max_,max_n])
    # comment start
    # convert categorical to numerical
    #comment end
    obj= obj.apply(u.convert,axis='columns')
    obj_n= obj_n.apply(u.convert,axis='columns')
    
    # comment start
    # impute the missing values
    #comment end
    impute_=u.inpute(numerical)
    impute_n=u.inpute(numerical_n)
    numerical_category=numerical.columns.values.tolist()
    
    impute_des=impute_.describe()
    impute_n_des=impute_n.describe()
    num_des=numerical.describe()
    
    # comment start
    # f1!!!!!!
    #comment end
    acc_f1=True
    if(acc_f1):
        f1=[]
        shape=[]
        for col in numerical_category:
            col=np.append(col,labels)
            df_col=df_all[col]
            df_col=df_col.dropna(axis='rows') 
            shape.append( df_col.shape[0])
            f1.append(u.treeClassifer(df_col,'churn'))
        np.array(f1).dump(open('f1.npy','wb'))
        print np.load(open('f1.npy','rb'))
        open('f1.txt','w').close()
        np.savetxt('f1.txt',f1)
        np.array(shape).dump(open('shape.npy','wb'))
        open('shape.txt','w').close()
        np.savetxt('shape.txt',shape)
        print np.loadtxt('f1.txt') , np.loadtxt('shape.txt') 