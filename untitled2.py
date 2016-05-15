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
    if (len(category)!=0):
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
    train=pd.concat([dfd,dfd_n.sample(dfd.shape[0])])
    category.extend(labels)
    train=train[category]
    train_des=train.describe()
    # comment start
    # classication
    #comment end
#    u.GNBClassifier(dfd,'churn')
#    dfd=pd.concat([dfd[dfd.columns.values[fea]],dfd[labels]],axis='columns')
#    f1=u.treeClassifer(train,'churn')
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