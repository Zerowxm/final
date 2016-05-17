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
        np.savetxt('f1_obj.txt',f1)
        np.savetxt('shape_obj.txt',shape)
        
        positive=[]
    to_acc=False
    if(to_acc):
        for col in numerical_category:
            col=np.append(col,label)
            df_col=df_all[col]
            positive.append(df_col.dropna(axis='rows')[df_all[label]==1].shape[0])
        np.savetxt('positive.txt',positive)
        
    positive=np.loadtxt('positive.txt')
    f1=np.loadtxt('f1.txt')
    f1_df=pd.DataFrame(f1,columns=['f1','accuracy','recall','roc_auc'])
    shape=np.loadtxt('shape.txt')
#    f1_df['count']=pd.Series(shape)
    f1_df['count_ratio']=pd.Series(shape)/df_all.shape[0]
    f1_df['positive']=pd.Series(positive)
    f1_df['positive_ratio']=pd.Series(positive).divide(shape)
    f1_df=f1_df.transpose()
    f1_df.columns=numerical_category
    
#    f1_df.transpose().plot()
#    plt.show()
#    f1=np.append(f1,shape)
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
    divide=False
    if(divide):
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
    dfd=pd.concat([dfd,classes],axis='columns')
    
    dfd_n=pd.concat([dfd_n,classes_n],axis='columns')
    train=pd.concat([dfd,dfd_n])
    train_features=train[category].apply(u.convert,axis='columns')
    
    train[category]=train_features
    
    category.extend(labels)
    train=train[category]
    X=train.as_matrix()
    temp_=temp.append(f1_df)
#    temp_=pd.concat([temp,f1_df],axis='rows')
#    smote=u.SMOTE(train[train.churn==1].as_matrix(),100,3)
#    sample_weight = np.array([5 if i == 0 else 1 for i in y])
#    sample_weight = [0 if x == -1 else 100 for x in train.churn  ]
#    train_des=train.describe()
    # comment start
    # classication
    #comment end
#    u.GNBClassifier(train,'churn')
#    u.treeClassifer(train,'churn')
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