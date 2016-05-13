# -*- coding: utf-8 -*-
"""
Created on Thu May 12 22:00:47 2016

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
def loadData(file,header=0):
    df=pd.read_table(file,header=header)
    return df
    
def cat(obj):
   for col in obj.columns:
         obj[col]=obj[col].astype('category')    
   return obj
   
def plot_rfe(X,label):
    y=X[label]
    X=X.drop(['churn','appetency','upselling',label],axis='columns')
    from sklearn.svm import SVC
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.feature_selection import RFECV
    import sklearn.linear_model as lm
    logreg = lm.LogisticRegression()
    clf = tree.DecisionTreeClassifier()
    # Build a classification task using 3 informative features
#    X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
#                               n_redundant=2, n_repeated=0, n_classes=8,
#                               n_clusters_per_class=1, random_state=0)
    
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=clf, step=1, cv=2,
                  scoring='accuracy')
                  
    print 'save'
    joblib.dump(rfecv, 'rfecv.pkl') 
    print 'fit'
    rfecv.fit(X, y)
    print 'save'
    joblib.dump(rfecv, 'rfecv.pkl.pkl') 
    print("Optimal number of features : %d" % rfecv.n_features_)
    
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
   
if __name__ == "__main__":
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
    
    df=pd.concat([df,churn,appetency,upselling],axis=1)
    df=df.dropna(axis='columns',how='all')[df.churn==-1]
#    null=pd.isnull(df)
#    isnull=df.isnull().any()
#    plt.figure()
#    data.plot(kind='')
    num=df.isnull().sum().sort_values(ascending=True,kind='quicksort')
    num_=num[num<40000]
    num_per=df.notnull().sum(axis=1).sort_values(ascending=True,kind='quicksort')
    mean_f=num_per.mean()
    minn= num_per[num_per>=num_per.mean()]
    features=np.array(num.index.tolist())
    f=['Var126','Var29','Var130','Var201','Var90','Var192','Var138','Var113','Var74','Var13',
   'Var189','Var205','Var73','Var211','Var199','Var212','Var217','Var2','Var218','Var81',
   'churn','appetency','upselling']
    df=df[features]
#    df=df.iloc[minn.index,:].dropna(axis='columns',how='any')
    des=df.describe()
    des=des.loc['count']/df.shape[0]
    des=des[des>0.5]
    df=df.dropna(axis='columns',how='all')
    dfd=df
    
#    dfd_any=df.dropna(axis='columns',how='any')
    obj=dfd.select_dtypes(include=['object'])
#    obj_any=dfd.select_dtypes(include=['object'])
#    catagory_any=obj.columns.values
    numerical=dfd.select_dtypes(exclude=['object'])
    impute_=u.inpute(numerical)
    impute_des=impute_.describe()
    num_des=numerical.describe()
    catagory=obj.columns.values
    numerical_catagory=impute_.columns.values
    obj= obj[catagory].apply(u.convert,axis='columns')
#    obj=cat(obj)
#    obj=obj.apply(u.inpute)
#    dfd.fillna(method='bfill')
    
    
#    print obj.isnull().sum().sum()
    dfd.loc[:,catagory]=dfd[catagory].apply(u.convert,axis='columns')

    dfd.loc[:,numerical_catagory]=impute_
#    u.cluster(dfd,'appetency')
#    des=dfd.describe()
#    mean=dfd.mean()
#    std=dfd.std()
#    norm=u.normalize_df(dfd)
#    dfd=(dfd-mean)/std
#    dfd=cat(dfd)
#    top=dfd[f]
#    top_des=top.describe()
#    top=u.inpute(top)
#    u.LRClassifer(dfd,'appetency')
#    u.treeClassifer(dfd,'appetency')
#    u.classification(dfd,'churn')
#    plot_rfe(dfd,'appetency')
#    d=u.selectFeaturesThres(dfd)
    cate = obj.describe()
    cate1=dfd.describe()
#    result=cat1.apply(pd.value_counts)
#    des=cat1.describe()
#    all_l=obj.dropna(axis='columns',how='any').columns.tolist()
#    corr=cat1.corr()
#    full=dfd.dropna(axis='columns',how='any')
#    u.treeClassifer(full,'appetency')
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