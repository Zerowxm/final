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
    
    label='churn'
    labels=['churn','appetency','upselling']
    temp=df.drop(df.dropna(axis='columns',how='any').columns.values,axis='columns').columns.values.tolist()
    null=pd.isnull(df[temp]).as_matrix().astype(np.int)
    missing_count=df.notnull().sum(axis=1)
    temp=[s + '_' for s in temp]
    null=pd.DataFrame(null,columns=temp)
#    df=pd.concat([df,null],axis='columns')
    # comment start
    # divide the data into two classes
    #comment end
    
    df_all=df.copy()
    
    df_all_cat=cat(df_all).describe()
    df_all_cat=df_all_cat.transpose()
    df_all_cat=df_all_cat[df_all_cat.unique>1]
#    df=df[df_all_cat.index]
    df_all_=df_all[df_all_cat.index]
    df_des=df.describe()
    
    
    df_rank=df.rank() # rank the columns
    df_=df[df[label]==-1] # choose the label == -1
    df=df[df[label]==1]# choose the label == 1
#    test(df_all)
       
    
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
    df_=df_.reindex(num_per_n.index)
#    num_per_n=num_per_n[num_per_n<num_per.min()]
    
    min_sam=df_.iloc[num_per_n]
    mean_f=num_per.mean() 
    minn= num_per[num_per>=num_per.mean()]
    features=np.array(num.index.tolist())
#    f=['Var126','Var29','Var130','Var201','Var90','Var192','Var138','Var113','Var74','Var13',
#   'Var189','Var205','Var73','Var211','Var199','Var212','Var217','Var2','Var218','Var81',
#   'churn','appetency','upselling']
   
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
    
    is_selected=False
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
    
    numerical_cate=cat(numerical).describe().transpose()
    numerical_n_cate=cat(numerical_n).describe().transpose()
    numerical_cate['type']=pd.Series(numerical.dtypes)
    numerical_n_cate['type']=pd.Series(numerical_n.dtypes)
    numerical_des=numerical.describe()
    numerical_n_des=numerical_n.describe()
#    numerical_unique=numerical.iloc[:,0].unique()
#    var=u.normalize_df(numerical).var()
#    var_n=u.normalize_df(numerical_n).var() 
    bug=False
    if(bug):
        count=numerical_cate['count']/numerical.shape[0]
        count.name='count_'
        count_n=numerical_n_cate['count']/numerical_n.shape[0]
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
        temp=pd.DataFrame([count,count_n,var,var_n,std,std_n,mean,mean_n,min_,min_n,max_,max_n]).transpose()
        temp_s=u.standardize_df(temp)
        
    # comment start
    # convert categorical to numerical
    #comment end
#    obj.fillna('0')
#    obj_n.fillna('0')
#    obj= obj.apply(u.convert_test,axis='index')
#    obj_n= obj_n.apply(u.conver_test,axis='index')
#    
    # comment start
    # impute the missing values
    #comment end
#    impute_=u.inpute(numerical)
#    impute_n=u.inpute(numerical_n)
    numerical_category=numerical.columns.values
#    
#    impute_des=impute_.describe()
#    impute_n_des=impute_n.describe()
    num_des=numerical.describe()
    
    
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
        temp_a=pd.concat([temp,f1_df],axis='columns')
        temp_a_s=u.standardize_df(temp_a)
        
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
    
#    f1_df_s.plot()
#    f1_obj_df_s.plot()
#    plt.show()
    
#    dfd.fillna(method='bfill')
    
#    dfd_des_b=dfd.describe()
#    dfd_n_des_b=dfd_n.describe()
    
#    diff= list(set(obj.columns.values).intersection(
#    set(impute_.columns.values))) 
    # comment start
    # replace the data by the preprocessed one
    #comment end
    divide=False
    if(divide):
        if (len(category)!=0):
            dfd_n[category]=obj_n
            dfd[category]=obj
#    top=dfd[f]
#    top_des=top.describe()
#    top=u.inpute(top)

#    dfd=u.selectFeaturesThres(dfd)    
      
    
    if(bug):
        dfd_des=dfd.describe()
        dfd_n_des=dfd_n.describe()
#    dfd_des.plot()
#    plt.figure()
#    dfd_n_des.plot()
    # comment start
    # get the train data
    #comment end
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
#    X=train.as_matrix()
    
#    temp_=pd.concat([temp,f1_df],axis='rows')
#    smote=u.SMOTE(train[train.churn==1].as_matrix(),100,3)
#    sample_weight = np.array([5 if i == 0 else 1 for i in y])
#    sample_weight = [0 if x == -1 else 100 for x in train.churn  ]
#    train_des=train.describe()
    # comment start
    # classication
    #comment end
#    u.GNBClassifier(train,'churn')
#    u.test(train,'churn')
#    dfd=pd.concat([dfd[dfd.columns.values[fea]],dfd[labels]],axis='columns')
#    f1=u.svmClassifer(train,'churn',sample_weight=sample_weight,cv=5)
    
#    u.treeRegression(train,'churn')
#    df_tmp=df_tmp[df_tmp.columns.values[bf]]
#    print df_tmp.describe()
#    df_tmp1=df_tmp1[df_tmp1.columns.values[bf]]
#    print df_tmp1.describe()
#    t=dfd.loc[:,labels]    
#    t1=dfd.drop(['churn','appetency','upselling'],axis='columns')[dfd.columns.values[bf]]
#    train=pd.concat([t1,t],axis='columns')
    
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
           
    
 