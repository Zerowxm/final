from sklearn import cross_validation
from collections import Counter
from scipy.stats import mode
import utils as u
import numpy  as np
import pandas as pd
#print Counter(y_train)
#ar=arr.argsort()[:3]
#df_test=df[category_del]
#median=numerical.median()
#numerical_fill=numerical.fillna(median,axis='rows')
#mode=mode(numerical)
#obj_all=obj_all.apply(u.convert_train,axis='index')
#print df_all.isnull().sum().sum()
f1=np.loadtxt('f1.txt')
#features=df_all.drop(labels,axis='columns').columns.values
f1_df=pd.DataFrame(f1,columns=['f1','accuracy','precision','recall','negative_a','positive_a'])
#f1_df=f1_df[f1_df.f1!=0]
features_index=f1_df.index.values


features_selected=features[features_index].tolist()
#features_selected.extend(labels)
features_selected.extend(features_categorical)
#features_selected=features_selected[:20]
#features_selected.extend(labels)
#test=df_all[features_selected]
#u.treeClassifer(test,'churn')
#features_selected=features[features_index].tolist()
#print Counter(f1_df['recall'])