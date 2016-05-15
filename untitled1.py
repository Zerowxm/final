# -*- coding: utf-8 -*-
"""
Created on Sat May 14 15:31:11 2016

@author: Zero
"""

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