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
    
    df_rank=df.rank() # rank the columns
    df_=df[df.churn==-1] # choose the label == -1
#    df=df[df.churn==1]# choose the label == 1
   
#    null=pd.isnull(df) 
#    isnull=df.isnull().any()
    
    
    # comment start
    # sort the features by missing values'count
    #comment end
    num=df.isnull().sum().sort_values(ascending=True,kind='quicksort') # sort the columns by the missing values' count
    thres=40000    
    num_=num[num<thres] # select data by shres
    num_per=df.notnull().sum(axis=1).sort_values(ascending=True,kind='quicksort') # every sample's  missing count
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
    a=0.9
    des_p_count=des_p.loc['count']/df.shape[0]
    des_p_count=des_p_count[des_p_count>a]
    des_n_count=des_n.loc['count']/df_.shape[0]
    des_n_count=des_n_count[des_n_count>a]
    des_diff= list(set(des_p_count.index.values).difference(
    set(des_n_count.index.values))) # diff the two classes features
    des_p=des_p[des_diff]
    
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
    
    # comment start
    # preprocess the data
    #comment end

#    dfd_any=df.dropna(axis='columns',how='any')
    
    # comment start
    # extract the string features
    #comment end
    obj=dfd.select_dtypes(include=['object'])
    obj_n=dfd_n.select_dtypes(include=['object'])
    category=obj.columns.values # string features' names
    
    # comment start
    # extract the numerical feautures
    #comment end
    numerical=dfd.select_dtypes(exclude=['object'])
    numerical_n=dfd_n.select_dtypes(exclude=['object'])
#    var=u.normalize_df(numerical).var()
#    var_n=u.normalize_df(numerical_n).var() 
    var=numerical.var()
    std=numerical.std()
    std.name='std'
    var.name='var'
    mean=numerical.mean()
    mean.name='mean'
    mean_n=numerical_n.mean()
    mean_n.name='mean_n'
    var_n=numerical_n.var()  
    std_n=numerical_n.std()
    std_n.name='std_n'
    var_n.name='var_n'
    temp=pd.DataFrame([var,var_n,std,std_n,mean,mean_n])
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
    numerical_category=impute_.columns.values
    
    impute_des=impute_.describe()
    impute_n_des=impute_n.describe()
    num_des=numerical.describe()
   
    
#    obj=cat(obj)
#    obj=obj.apply(u.inpute)
#    dfd.fillna(method='bfill')
    
#    print obj.isnull().sum().sum()
    
    dfd_des_b=dfd.describe()
    dfd_n_des_b=dfd_n.describe()
    
#    diff= list(set(obj.columns.values).intersection(
#    set(impute_.columns.values))) 
    # comment start
    # replace the data by the preprocessed one
    #comment end
    if (category.shape[0]!=0):
        dfd_n[category]=obj_n
        dfd[category]=obj
        dfd=u.inpute(dfd)
        dfd_n=u.inpute(dfd_n)
    else:
        dfd=impute_
        dfd_n=impute_n
#    top=dfd[f]
#    top_des=top.describe()
#    top=u.inpute(top)

#    dfd=u.selectFeaturesThres(dfd)    
      
    
    dfd_des=dfd.describe()
    dfd_n_des=dfd_n.describe()
#    dfd_des.plot()
#    plt.figure()
#    dfd_n_des.plot()
    # comment start
    # get the train data
    #comment end
    train=pd.concat([dfd,dfd_n])
    train_des=train.describe()
    # comment start
    # classication
    #comment end
#    u.GNBClassifier(dfd,'churn')
#    dfd=pd.concat([dfd[dfd.columns.values[fea]],dfd[labels]],axis='columns')
#    bf=u.treeClassifer(dfd,'churn')
#    df_tmp=df_tmp[df_tmp.columns.values[bf]]
#    print df_tmp.describe()
#    df_tmp1=df_tmp1[df_tmp1.columns.values[bf]]
#    print df_tmp1.describe()
    t=dfd.loc[:,labels]    
    t1=dfd.drop(['churn','appetency','upselling'],axis='columns')[dfd.columns.values[bf]]
    train=pd.concat([t1,t],axis='columns')
#    u.treeClassifer(train,'churn')
#    u.classification(train,'churn')
#    plt.matshow(train.corr())
#    beta2 = (train.corr() * df['b'].std() * df['a'].std() / df['a'].var()).ix[0, 1]
#    print(beta2)
#    u.cluster(train,'appetency')
#    result=cat1.apply(pd.value_counts)
#    all_l=obj.dropna(axis='columns',how='any').columns.tolist()
#    corr=cat1.corr()
#    full=dfd.dropna(axis='columns',how='any')
#    u.treeClassifer(train,'churn')
#    column='Var202'
#    test=obj[pd.isnull(df[column])][all_l]
#    full=obj[pd.notnull(df[column])]
#    g=full.groupby('Var220')
#    train=full[all_l]
#    df_num=train.apply(u.convert,axis='columns')
#    d=u.selectFeaturesThres(df_num)
#    label=full[column]
#    label_num=u.convert(label)
    
#    train_dict=train.to_dict()
#    dict_val=train_dict.values()
#    u.classification(df_num,label)
#    runFp(obj)
#    runAproiri(obj.values.tolist(),minsup=0.5,minconf=0.5)
#    cat.plot().line()
#    df=df[]
#    plt.title('missing values')
#    num.plot()