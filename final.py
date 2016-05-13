
# coding: utf-8

# In[3]:

import apriori as ap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fpGrowth as fp
import utils as u


# In[4]:

def loadData(file,header=0):
    df=pd.read_table(file,header=header)
    return df


# In[5]:

def convert_category(obj):
   for col in obj.columns:
         obj[col]=obj[col].astype('category')    
   return obj


# In[6]:




# In[7]:

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


# In[8]:

df=pd.concat([df,churn,appetency,upselling],axis=1)


# In[9]:

#    null=pd.isnull(df)
#    isnull=df.isnull().any()
#    plt.figure()
#    data.plot(kind='')


# In[10]:

num=df.isnull().sum().sort_values(ascending=True,kind='quicksort')
num1=df.isnull().sum(axis='rows').sort_values(ascending=True,kind='quicksort')
features=np.array(num.index.tolist())


# In[11]:

f=['Var126','Var29','Var130','Var201','Var90','Var192','Var138','Var113','Var74','Var13',
   'Var189','Var205','Var73','Var211','Var199','Var212','Var217','Var2','Var218','Var81',
   'churn','appetency','upselling']


# In[12]:

dfd=df.dropna(axis='columns',how='all')
dfd=convert_category(dfd)
cat1=dfd.describe()


# In[ ]:

#    dfd.fillna(method='bfill')


# In[15]:

obj=dfd.select_dtypes(include=['object'])
#    print obj.isnull().sum().sum()
obj=convert_category(obj)
#    obj=obj.apply(u.inpute)
category=obj.columns.values
cat = obj.describe()


# In[ ]:

dfd[catagory]=dfd[catagory].apply(u.convert,axis='columns')
top=dfd[f]
top=u.inpute(top)
u.treeClassifer(top)


# In[ ]:

#    d=u.selectFeaturesThres(dfd)
#    result=cat1.apply(pd.value_counts)
#    des=cat1.describe()

#    corr=cat1.corr()
#    full=dfd.dropna(axis='columns',how='any')
    all_l=obj.dropna(axis='columns',how='any').columns.tolist()


# In[ ]:

column='Var202'
test=obj[pd.isnull(df[column])][all_l]
full=obj[pd.notnull(df[column])]
#    g=full.groupby('Var220')
train=full[all_l]
df_num=train.apply(u.convert,axis='columns')
#    d=u.selectFeaturesThres(df_num)
label=full[column]
label_num=u.convert(label)


# In[ ]:

#    train_dict=train.to_dict()
#    dict_val=train_dict.values()
#    u.classification(df_num,label)
#    runFp(obj)
#    runAproiri(obj.values.tolist(),minsup=0.5,minconf=0.5)
#    cat.plot().line()
#    df=df[]
#    plt.title('missing values')
#    num.plot()


# In[ ]:



