# -*- coding: utf-8 -*-
"""
    Created on Sat Apr 16 18:14:53 2016

@author: Zero
"""
from random import  choice
from sklearn.feature_selection import RFE
from sklearn.neighbors import NearestNeighbors
from time import time
from functools import wraps
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile, f_classif
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier,BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold,SelectFromModel
from sklearn.preprocessing import Imputer
from sklearn import tree
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc,roc_auc_score,mean_squared_error
import sklearn.linear_model as lm
from sklearn.neighbors import KNeighborsClassifier
from itertools import cycle
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 
from sklearn import svm   
from collections import Counter
from sklearn.metrics import cohen_kappa_score, make_scorer
from skll.metrics import kappa
from sklearn.grid_search import ParameterGrid 
from sklearn.utils.testing import assert_true
from sklearn import linear_model  
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import make_pipeline 
from unbalanced_dataset.over_sampling import OverSampler
from unbalanced_dataset.over_sampling import SMOTE
from unbalanced_dataset.under_sampling import UnderSampler
from unbalanced_dataset.under_sampling import TomekLinks
from unbalanced_dataset.under_sampling import ClusterCentroids
from unbalanced_dataset.under_sampling import NearMiss
from unbalanced_dataset.under_sampling import CondensedNearestNeighbour
from unbalanced_dataset.under_sampling import OneSidedSelection
from unbalanced_dataset.under_sampling import NeighbourhoodCleaningRule
from unbalanced_dataset.ensemble_sampling import EasyEnsemble
from unbalanced_dataset.ensemble_sampling import BalanceCascade
from unbalanced_dataset.pipeline import SMOTEENN
from unbalanced_dataset.pipeline import SMOTETomek
plt.style.use('fivethirtyeight') # Good looking plots
pd.set_option('display.max_columns', None) # Display any number of columns
    
def blend(X_train,y_train,X_test,y_test):
    np.random.seed(0) # seed to shuffle the train set
    n_folds = 10
    verbose = True
    shuffle = False
    X=X_train
    y=y_train
    X_submission=X_test

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(cross_validation.StratifiedKFold(y, n_folds))
    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            DecisionTreeClassifier(),
            AdaBoostClassifier(n_estimators=100, base_estimator=DecisionTreeClassifier(),learning_rate=1),
            GradientBoostingClassifier( subsample=0.5, max_depth=6, n_estimators=50)]

    print "Creating train and test sets for blending."
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    
    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    print "Blending."
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_pred = clf.predict(dataset_blend_test)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    print "Linear stretch of predictions to [0,1]"
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print "Saving Results."
    np.savetxt(fname='test.csv', X=y_submission, fmt='%0.9f')  
def cutoff_predict(clf,X,cutoff):
    return (clf.predict_proba(X)[:,1]>cutoff).astype(int)

def custom_f1(cutoff):
    def f1_cutoff(clf,X,y):
        ypred= cutoff_predict(clf,X,cutoff)
        return metrics.f1_score(y,ypred)
    return f1_cutoff
   
def test(X_train,y_train,X_test,y_test, sample_weight=None,cv=10):
    for cutoff in np.arange(0.1,0.9,0.05):
        clf=RandomForestClassifier(n_estimators=15,class_weight ='balanced') 
        clf = GradientBoostingClassifier()
        clf.fit(X_train,y_train)
#        scores(y_test,cutoff_predict(clf,X_test,cutoff),'gbc')
        valideted=cross_validation.cross_val_score(clf,X_train,y_train,scoring=custom_f1(cutoff))
        print(" f1: %0.2f (+/- %0.2f)" % (valideted.mean(), valideted.std()))
def abClassifier(X_train,y_train,X_test,y_test,to_plot=False):
    params = {
          'random_state': [None,0,1,2,3,4,5]}
    for param in ParameterGrid(params):
        print param
        clf = AdaBoostClassifier(algorithm='SAMME.R',n_estimators=50,learning_rate=0.7,**param)
        clf.fit(X_train,y_train)
#        auc_compute(y_test,clf.predict_proba(X_test)[:,1])
        predictions=clf.predict(X_test)
        scores(y_test,predictions,clf.predict_proba(X_test)[:,1],'ab',to_plot=to_plot)   
        
def gbClassifier(X_train,y_train,X_test,y_test,to_plot=False):
    params = {
          'subsample': [0.5]}
    for param in ParameterGrid(params):
        print param
        clf = GradientBoostingClassifier(max_depth=7,n_estimators=50,learning_rate=0.05,subsample=0.5)
        clf.fit(X_train,y_train)
#        print roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
#        auc_compute(y_test,clf.predict_proba(X_test)[:,1])
#    bag=BaggingClassifier()
#    bag.fit(X_train,y_train)
#    predictions=bag.predict(X_test)
#    scores(y_test,predictions,'bag',to_plot=to_plot)  
        predictions=clf.predict(X_test)
        scores(y_test,predictions,clf.predict_proba(X_test)[:,1],'gbc',to_plot=to_plot)   
#    mse=mean_squared_error(y_test, predictions)
#    print("MSE: %.4f" % mse)
#    joblib.dump(clf,'gbc2.pkl')
#    return scores(y_test,predictions,'gbc',to_plot=to_plot)    
def cross_val(X_train,y_train,to_plot=False):
    scores(y_train,stratified_cv(X_train,y_train,GradientBoostingClassifier,n_folds=5),'gbc',to_plot=to_plot)   

def all_classifer(X_train,y_train,X_test,y_test):
    rf=RandomForestClassifier(n_estimators=100,class_weight ='balanced') 
    score1=scores(y_test,rf.fit(X_train,y_train).predict(X_test),rf.predict_proba(X_test)[:,1],'RT')
    gbc = GradientBoostingClassifier(n_estimators=50,learning_rate=0.05).fit(X_train,y_train)
    score2=scores(y_test,gbc.fit(X_train,y_train).predict(X_test),gbc.predict_proba(X_test)[:,1],'gbc') 
    ets=ExtraTreesClassifier(n_estimators=100,max_depth=None,min_samples_split=1,random_state=0)
    score3=scores(y_test,ets.fit(X_train,y_train).predict(X_test),ets.predict_proba(X_test)[:,1],'ets') 
#    lgr = LogisticRegression()
#    score4=scores(y_test,lgr.fit(X_train,y_train).predict(X_test),'lgr') 
    ab = AdaBoostClassifier(algorithm='SAMME.R',n_estimators=50,learning_rate=0.7)
    score5=scores(y_test,ab.fit(X_train,y_train).predict(X_test),ab.predict_proba(X_test)[:,1],'abboost') 
#    print roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
#    bagging=BaggingClassifier()
#    score8=scores(y_test,bagging.fit(X_train,y_train).predict(X_test),'bagging')    
    
#    dt = DecisionTreeClassifier(max_depth=None, min_samples_split=1,random_state=0)
#    score6=scores(y_test,dt.fit(X_train,y_train).predict(X_test),'dt') 
    eclf = VotingClassifier(estimators=[ ('rf', rf), 
                                        ('gd',gbc),('ETs',ets),('ab',ab)],
                                         voting='soft',weights =[score1[0],score2[0],score3[0],score5[0]])
    score7=scores(y_test,eclf.fit(X_train,y_train).predict(X_test),eclf.predict_proba(X_test)[:,1],'voting') 
    print eclf
    return [score1,score2,score3,score5,score7]
def stackingClassifier(X_train,y_train,X_test,y_test):
    base_algorithms =[AdaBoostClassifier(n_estimators=100),
                      RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
                    RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
                    ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
                    GradientBoostingClassifier( subsample=0.5, max_depth=6, n_estimators=50)]    
    stacking_train_dataset = np.zeros((y_train.shape[0], len(base_algorithms)))
    stacking_test_dataset = np.zeros((y_test.shape[0], len(base_algorithms)))
    for i,base_algorithm in enumerate(base_algorithms):
        stacking_train_dataset[:,i] = base_algorithm.fit(X_train, y_train).predict(X_train)
        stacking_test_dataset[:,i] = base_algorithm.predict(X_test)
        scores(y_test,stacking_test_dataset[:,i] ,str(i))
    return stacking_train_dataset,stacking_test_dataset     
import os
import seaborn as sns
def cof(X,y):
    pass_agg_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, linear_model.PassiveAggressiveClassifier))
    grad_ens_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, GradientBoostingClassifier))
    decision_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, tree.DecisionTreeClassifier))
    ridge_clf_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, linear_model.RidgeClassifier))
    svm_svc_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, svm.SVC))
    random_forest_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y,RandomForestClassifier))
    k_neighbors_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, KNeighborsClassifier))
    logistic_reg_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, linear_model.LogisticRegression))
    dumb_conf_matrix = metrics.confusion_matrix(y, [0 for ii in y.tolist()]); # ignore the warning as they are all 0
    conf_matrix = {
                    1: {
                        'matrix': pass_agg_conf_matrix,
                        'title': 'Passive Aggressive',
                       },
                    2: {
                        'matrix': grad_ens_conf_matrix,
                        'title': 'Gradient Boosting',
                       },
                    3: {
                        'matrix': decision_conf_matrix,
                        'title': 'Decision Tree',
                       },
                    4: {
                        'matrix': ridge_clf_conf_matrix,
                        'title': 'Ridge',
                       },
                    5: {
                        'matrix': svm_svc_conf_matrix,
                        'title': 'Support Vector Machine',
                       },
                    6: {
                        'matrix': random_forest_conf_matrix,
                        'title': 'Random Forest',
                       },
                    7: {
                        'matrix': k_neighbors_conf_matrix,
                        'title': 'K Nearest Neighbors',
                       },
                    8: {
                        'matrix': logistic_reg_conf_matrix,
                        'title': 'Logistic Regression',
                       },
                    9: {
                        'matrix': dumb_conf_matrix,
                        'title': 'Dumb',
                       },
    }
    fix, ax = plt.subplots(figsize=(16, 12))
    plt.suptitle('Confusion Matrix of Various Classifiers')
    for ii, values in conf_matrix.items():
        matrix = values['matrix']
        title = values['title']
        plt.subplot(3, 3, ii) # starts from 1
        plt.title(title);
        sns.heatmap(matrix, annot=True,  fmt='');
        
def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);


def featureImp(X,y,forest):
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
#     Print the feature ranking
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

def stratified_cv(X, y, clf_class, shuffle=True, n_folds=10, **kwargs):
    stratified_k_fold = cross_validation.StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
    y_pred = y.copy()
    for ii, jj in stratified_k_fold:
        X_train, X_test = X[ii], X[jj]
        y_train = y[ii]
#        y_test = y[jj]
#        print Counter(y_train)
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[jj] = clf.predict(X_test)
#        print 'test' ,Counter(y_test)
    return y_pred
    
def k_fold_generator(X, y, k_fold):
    subset_size = len(X) / k_fold  # Cast to int if using Python 3
    for k in range(k_fold):
        X_train = X[:k * subset_size] + X[(k + 1) * subset_size:]
        X_valid = X[k * subset_size:][:subset_size]
        y_train = y[:k * subset_size] + y[(k + 1) * subset_size:]
        y_valid = y[k * subset_size:][:subset_size]

        yield X_train, y_train, X_valid, y_valid
        
# Utility function to report best scores
def report(grid_scores, n_top=3):
    from operator import itemgetter
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
    
def inpute_category(X):
    c=X.columns
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    return pd.DataFrame(imp.fit_transform(X),columns=c)

def intersection(a,b):
    return list(set(a).intersection(set(b)))
def intersection_count(a,b):
    unique=list(set(a).intersection(set(b)))
    return float(len(unique))/len(a)    
def normalize_df(X):
    c=X.columns
    i=X.index
    normalized_X = preprocessing.normalize(X)
    return pd.DataFrame(normalized_X,index=i,columns=c)    
def standardize_df(X):
    c=X.columns
    i=X.index
    X=X.as_matrix()
    scaler=preprocessing.StandardScaler()
    standardized_X = scaler.fit_transform(X)
    return pd.DataFrame(standardized_X,index=i,columns=c)

def fn_timer(function):
  @wraps(function)
  def function_timer(*args, **kwargs):
    t0 = time()
    result = function(*args, **kwargs)
    t1 = time()
    print ("Total time running %s: %s seconds" %
        (function.func_name, str(t1-t0))
        )
    return result
  return function_timer

verbose = True
indices_support = True
def test_smote(x, y,c=0):
    if(c==0):
        print('SMOTE')
        sm = SMOTE(kind='regular', verbose=verbose)
        svmx, svmy = sm.fit_transform(x, y)
    elif(c==1):
        print('SMOTE bordeline 1')
        sm = SMOTE(kind='borderline1', verbose=verbose)
        svmx, svmy = sm.fit_transform(x, y)
    elif(c==2):
        print('SMOTE bordeline 2')
        sm = SMOTE(kind='borderline2', verbose=verbose)
        svmx, svmy = sm.fit_transform(x, y)
    elif(c==3):
        print('SMOTE SVM')
        svm_args={'class_weight': 'auto'}
        sm = SMOTE(kind='svm', verbose=verbose, **svm_args)
        svmx, svmy = sm.fit_transform(x, y)
    return svmx,svmy


def test_rest(x, y,c=0,ratio='auto'):
    c=c
    if(c==0):
        print('Random under-sampling')
        US = UnderSampler(indices_support=indices_support, verbose=verbose,ratio=ratio)
        x, y, idx_tmp = US.fit_transform(x, y)
        print ('Indices selected')
        print(idx_tmp)
    elif(c==1):
        print('Tomek links')
        TL = TomekLinks(verbose=verbose,ratio=ratio)
        x, y = TL.fit_transform(x, y)
    elif(c==2):
        print('Clustering centroids')
        CC = ClusterCentroids(verbose=verbose,ratio=ratio)
        x, y = CC.fit_transform(x, y)
    elif(c==3):
        print('NearMiss-1')
        NM1 = NearMiss(version=1, indices_support=indices_support, verbose=verbose,ratio=ratio)
        x, y, idx_tmp = NM1.fit_transform(x, y)
        print ('Indices selected')
        print(idx_tmp)
    elif(c==4):
        print('NearMiss-2')
        NM2 = NearMiss(version=2, indices_support=indices_support, verbose=verbose,ratio=ratio)
        x, y, idx_tmp = NM2.fit_transform(x, y)
        print ('Indices selected')
        print(idx_tmp)
    elif(c==5):
        print('NearMiss-3')
        NM3 = NearMiss(version=3, indices_support=indices_support, verbose=verbose,ratio=ratio)
        x, y, idx_tmp = NM3.fit_transform(x, y)
        print ('Indices selected')
        print(idx_tmp)
    elif(c==6):
        print('Neighboorhood Cleaning Rule')
        NCR = NeighbourhoodCleaningRule(indices_support=indices_support, verbose=verbose,ratio=ratio)
        x, y, idx_tmp = NCR.fit_transform(x, y)
        print ('Indices selected')
        print(idx_tmp)
    elif(c==7):
        print('Random over-sampling')
        OS = OverSampler(verbose=verbose,ratio=ratio)
        x, y = OS.fit_transform(x, y)
    elif(c==8):
        print('SMOTE Tomek links')
        STK = SMOTETomek(verbose=verbose,ratio=ratio)
        x, y = STK.fit_transform(x, y)
    elif(c==9):
        print('SMOTE ENN')
        SENN = SMOTEENN(verbose=verbose,ratio=ratio)
        x, y = SENN.fit_transform(x, y)
    else:
        print('EasyEnsemble')
        EE = EasyEnsemble(verbose=verbose,ratio=ratio)
        x, y = EE.fit_transform(x, y)
    return x, y

def test_CNN(x, y,c=0,ratio='auto'):
    if(c==0):
        print('Condensed Nearest Neighbour')
        CNN = CondensedNearestNeighbour(indices_support=indices_support, verbose=verbose)
        x,y, idx_tmp = CNN.fit_transform(x, y)
        print ('Indices selected')
        print(idx_tmp)
    elif(c==1):
        print('One-Sided Selection')
        OSS = OneSidedSelection(indices_support=indices_support, verbose=verbose)
        x,y, idx_tmp = OSS.fit_transform(x, y)
        print ('Indices selected')
        print(idx_tmp)
    elif(c==2):
        print('BalanceCascade')
        BS = BalanceCascade(ratio='auto',verbose=verbose)
        x,y = BS.fit_transform(x, y)
    return x,y  
import math, random, copy

def expectation_maximization(t, nbclusters=2, nbiter=3, normalize=False,
        epsilon=0.001, monotony=False, datasetinit=True):
    """ 
    Each row of t is an observation, each column is a feature 
    'nbclusters' is the number of seeds and so of clusters
    'nbiter' is the number of iterations
    'epsilon' is the convergence bound/criterium

    Overview of the algorithm:
    -> Draw nbclusters sets of (μ, σ, P_{#cluster}) at random (Gaussian 
       Mixture) [P(Cluster=0) = P_0 = (1/n).∑_{obs} P(Cluster=0|obs)]
    -> Compute P(Cluster|obs) for each obs, this is:
    [E] P(Cluster=0|obs)^t = P(obs|Cluster=0)*P(Cluster=0)^t
    -> Recalculate the mixture parameters with the new estimate
    [M] * P(Cluster=0)^{t+1} = (1/n).∑_{obs} P(Cluster=0|obs)
        * μ^{t+1}_0 = ∑_{obs} obs.P(Cluster=0|obs) / P_0
        * σ^{t+1}_0 = ∑_{obs} P(Cluster=0|obs)(obs-μ^{t+1}_0)^2 / P_0
    -> Compute E_t=∑_{obs} log(P(obs)^t)
       Repeat Steps 2 and 3 until |E_t - E_{t-1}| < ε
    """
    def pnorm(x, m, s):
        """ 
        Compute the multivariate normal distribution with values vector x,
        mean vector m, sigma (variances/covariances) matrix s
        """
        xmt = np.matrix(x-m).transpose()
        for i in xrange(len(s)):
            if s[i,i] <= sys.float_info[3]: # min float
                s[i,i] = sys.float_info[3]
        sinv = np.linalg.inv(s)
        xm = np.matrix(x-m)
        return (2.0*math.pi)**(-len(x)/2.0)*(1.0/math.sqrt(np.linalg.det(s)))\
                *math.exp(-0.5*(xm*sinv*xmt))

    def draw_params():
            if datasetinit:
                tmpmu = np.array([1.0*t[random.uniform(0,nbobs),:]],np.float64)
            else:
                tmpmu = np.array([random.uniform(min_max[f][0], min_max[f][1])\
                        for f in xrange(nbfeatures)], np.float64)
            return {'mu': tmpmu,\
                    'sigma': np.matrix(np.diag(\
                    [(min_max[f][1]-min_max[f][0])/2.0\
                    for f in xrange(nbfeatures)])),\
                    'proba': 1.0/nbclusters}

    nbobs = t.shape[0]
    nbfeatures = t.shape[1]
    min_max = []
    # find xranges for each features
    for f in xrange(nbfeatures):
        min_max.append((t[:,f].min(), t[:,f].max()))
    
    ### Normalization
    if normalize:
        for f in xrange(nbfeatures):
            t[:,f] -= min_max[f][0]
            t[:,f] /= (min_max[f][1]-min_max[f][0])
    min_max = []
    for f in xrange(nbfeatures):
        min_max.append((t[:,f].min(), t[:,f].max()))
    ### /Normalization

    result = {}
    quality = 0.0 # sum of the means of the distances to centroids
    random.seed()
    Pclust = np.ndarray([nbobs,nbclusters], np.float64) # P(clust|obs)
    Px = np.ndarray([nbobs,nbclusters], np.float64) # P(obs|clust)
    # iterate nbiter times searching for the best "quality" clustering
    for iteration in xrange(nbiter):
        ##############################################
        # Step 1: draw nbclusters sets of parameters #
        ##############################################
        params = [draw_params() for c in xrange(nbclusters)]
        old_log_estimate = sys.maxint         # init, not true/real
        log_estimate = sys.maxint/2 + epsilon # init, not true/real
        estimation_round = 0
        # Iterate until convergence (EM is monotone) <=> < epsilon variation
        while (abs(log_estimate - old_log_estimate) > epsilon\
                and (not monotony or log_estimate < old_log_estimate)):
            restart = False
            old_log_estimate = log_estimate
            ########################################################
            # Step 2: compute P(Cluster|obs) for each observations #
            ########################################################
            for o in xrange(nbobs):
                for c in xrange(nbclusters):
                    # Px[o,c] = P(x|c)
                    Px[o,c] = pnorm(t[o,:],\
                            params[c]['mu'], params[c]['sigma'])
            #for o in xrange(nbobs):
            #    Px[o,:] /= math.fsum(Px[o,:])
            for o in xrange(nbobs):
                for c in xrange(nbclusters):
                    # Pclust[o,c] = P(c|x)
                    Pclust[o,c] = Px[o,c]*params[c]['proba']
            #    assert math.fsum(Px[o,:]) >= 0.99 and\
            #            math.fsum(Px[o,:]) <= 1.01
            for o in xrange(nbobs):
                tmpSum = 0.0
                for c in xrange(nbclusters):
                    tmpSum += params[c]['proba']*Px[o,c]
                Pclust[o,:] /= tmpSum
                #assert math.fsum(Pclust[:,c]) >= 0.99 and\
                #        math.fsum(Pclust[:,c]) <= 1.01
            ###########################################################
            # Step 3: update the parameters (sets {mu, sigma, proba}) #
            ###########################################################
            print "iter:", iteration, " estimation#:", estimation_round,\
                    " params:", params
            for c in xrange(nbclusters):
                tmpSum = math.fsum(Pclust[:,c])
                params[c]['proba'] = tmpSum/nbobs
                if params[c]['proba'] <= 1.0/nbobs:           # restart if all
                    restart = True                             # converges to
                    print "Restarting, p:",params[c]['proba'] # one cluster
                    break
                m = np.zeros(nbfeatures, np.float64)
                for o in xrange(nbobs):
                    m += t[o,:]*Pclust[o,c]
                params[c]['mu'] = m/tmpSum
                s = np.matrix(np.diag(np.zeros(nbfeatures, np.float64)))
                for o in xrange(nbobs):
                    s += Pclust[o,c]*(np.matrix(t[o,:]-params[c]['mu']).transpose()*\
                            np.matrix(t[o,:]-params[c]['mu']))
                    #print ">>>> ", t[o,:]-params[c]['mu']
                    #diag = Pclust[o,c]*((t[o,:]-params[c]['mu'])*\
                    #        (t[o,:]-params[c]['mu']).transpose())
                    #print ">>> ", diag
                    #for i in xrange(len(s)) :
                    #    s[i,i] += diag[i]
                params[c]['sigma'] = s/tmpSum
                print "------------------"
                print params[c]['sigma']

            ### Test bound conditions and restart consequently if needed
            if not restart:
                restart = True
                for c in xrange(1,nbclusters):
                    if not np.allclose(params[c]['mu'], params[c-1]['mu'])\
                    or not np.allclose(params[c]['sigma'], params[c-1]['sigma']):
                        restart = False
                        break
            if restart:                # restart if all converges to only
                old_log_estimate = sys.maxint          # init, not true/real
                log_estimate = sys.maxint/2 + epsilon # init, not true/real
                params = [draw_params() for c in xrange(nbclusters)]
                continue
            ### /Test bound conditions and restart

            ####################################
            # Step 4: compute the log estimate #
            ####################################
            log_estimate = math.fsum([math.log(math.fsum(\
                    [Px[o,c]*params[c]['proba'] for c in xrange(nbclusters)]))\
                    for o in xrange(nbobs)])
            print "(EM) old and new log estimate: ",\
                    old_log_estimate, log_estimate
            estimation_round += 1

        # Pick/save the best clustering as the final result
        quality = -log_estimate
        if not quality in result or quality > result['quality']:
            result['quality'] = quality
            result['params'] = copy.deepcopy(params)
            result['clusters'] = [[o for o in xrange(nbobs)\
                    if Px[o,c] == max(Px[o,:])]\
                    for c in xrange(nbclusters)]
    return result
def to_category(obj):
   tmp=obj.copy()
   for col in tmp.columns:
         tmp[col]=tmp[col].astype('category')    
   return tmp
   
def randScore(est,labels_true):
    labels = est.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
def convert_train(x):
     le = preprocessing.LabelEncoder()
     new_x=le.fit_transform(x)
     filename =os.path.join('test','%sle.pkl'%x.name)
     joblib.dump(le,filename)
     return new_x
def convert_test(x):
     filename =os.path.join('test','%sle.pkl'%x.name)
     le = joblib.load(filename)
     new_x=le.transform(x)
     return new_x     
     
def inpute(X):
    c=X.columns
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    return pd.DataFrame(imp.fit_transform(X),columns=c)  
    
def selectFeaturesThres(X):
    c=X.columns
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    return pd.DataFrame(sel.fit_transform(X),columns=c)
 
def selectFeatures_train(X,y,k=20,p=20):
    selector = SelectPercentile(f_classif, percentile=p)
    selector.fit(X, y)
    joblib.dump(selector,'selector.pkl')
    select_best = SelectKBest(f_classif, k=k).fit(X, y)
    joblib.dump(select_best,'SelectKBest.pkl')
def selectFeatures(X,t=0):
    if(t==0):
        selector=joblib.load('selector.pkl')
    else:
        selector=joblib.load('SelectKBest.pkl')
    X_new=selector.transform(X)
    return X_new
def plot_confusion_matrix(cm,target, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target))
    plt.xticks(tick_marks,target, rotation=45)
    plt.yticks(tick_marks,target)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plt_cm(y_test, y_pred,target_names):
    cm = confusion_matrix(y_test, y_pred)
    print cm
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    plt.figure()
    plot_confusion_matrix(cm_normalized,target_names)
    plt.show()
  
def cluster(X,label):
    labels_true=X[label]
    X=X.drop(['churn','appetency','upselling',label],axis='columns')
    name='AffinityPropagation'
    est=AffinityPropagation(preference=-50)
    adjustD={}
    clusters_n={}
    db = est.fit(X)
    labels = db.labels_
    adjustD[name]=metrics.adjusted_rand_score(labels_true,labels)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    clusters_n[name]=n_clusters_
    print('Estimated estimator: %s' % name)
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
          
def plot_rfe(X,y):
    from sklearn.svm import SVC
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.feature_selection import RFECV
    logreg = lm.LogisticRegression()
    clf = tree.DecisionTreeClassifier()
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=clf, step=1, cv=2,
                  scoring='f1')
                  
    rfecv.fit(X, y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
def rank_to_dict(ranks, names, order=1):
    minmax = preprocessing.MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))
    
def smote(T, N, k):
    """
    Returns (N/100) * n_minority_samples synthetic minority samples.

    Parameters
    ----------
    T : array-like, shape = [n_minority_samples, n_features]
        Holds the minority samples
    N : percetange of new synthetic samples: 
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours. 

    Returns
    -------
    S : array, shape = [(N/100) * n_minority_samples, n_features]
    """    
    n_minority_samples, n_features = T.shape
    
    if N < 100:
        #create synthetic samples only for a subset of T.
        #TODO: select random minortiy samples
        N = 100
        pass

    if (N % 100) != 0:
        raise ValueError("N must be < 100 or multiple of 100")
    
    N = N/100
    n_synthetic_samples = N * n_minority_samples
    S = np.zeros(shape=(n_synthetic_samples, n_features))
    
    #Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(T)
    
    #Calculate synthetic samples
    for i in xrange(n_minority_samples):
        nn = neigh.kneighbors(T[i], return_distance=False)
        for n in xrange(N):
            nn_index = choice(nn[0])
            #NOTE: nn includes T[i], we don't want to select it 
            while nn_index == i:
                nn_index = choice(nn[0])
                
            dif = T[nn_index] - T[i]
            gap = np.random.random()
            S[n + i * N, :] = T[i,:] + gap * dif[:]
    
    return S     
    
def ploy_standard_scale(df):
    X = df.as_matrix().astype(np.float)
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    polynomial_features = preprocessing.PolynomialFeatures()
    X = polynomial_features.fit_transform(X)
    return X
def cross(est,X,y,name,cm=False,cv=10):
    kappa_scorer = make_scorer(cohen_kappa_score)
    print('%d-fold cross validation:\n'%cv)
    start_time = time()
    scores=[]
    f1 = cross_validation.cross_val_score(est, X,y, cv=10, scoring ='f1')
    scores.append(f1)
    print(" f1: %0.2f (+/- %0.2f) [%s]" % (f1.mean(), f1.std(), name))
    kappa_scorer = cross_validation.cross_val_score(est, X,y, cv=cv, scoring =kappa_scorer)
    print(" kappa_scorer: %0.2f (+/- %0.2f) [%s]" % (kappa_scorer.mean(), kappa_scorer.std(), name))
    accuracy = cross_validation.cross_val_score(est, X,y, cv=cv, scoring ='accuracy')
    scores.append(accuracy)
    print(" accuracy: %0.2f (+/- %0.2f) [%s]" % (accuracy.mean(), accuracy.std(), name))
    precision = cross_validation.cross_val_score(est, X,y, cv=cv, scoring ='precision')
    scores.append(precision)
    print(" precision: %0.2f (+/- %0.2f) [%s]" % (precision.mean(), precision.std(), name))
    recall = cross_validation.cross_val_score(est, X,y, cv=cv, scoring ='recall')
    scores.append(recall)
    print(" recall: %0.2f (+/- %0.2f) [%s]" % (recall.mean(), recall.std(), name))
    roc_auc = cross_validation.cross_val_score(est, X,y, cv=cv, scoring ='roc_auc')
    scores.append(roc_auc)
    print(" roc_auc: %0.2f (+/- %0.2f) [%s]" % (roc_auc.mean(), roc_auc.std(), name))
    print("---Classifier %s use %s seconds ---" %(name, (time() - start_time)))
    return [f1.mean(),accuracy.mean(),recall.mean(),precision.mean(),roc_auc.mean()]
    
def km(X,label):
    from sklearn.cluster import KMeans
    est=KMeans(n_clusters=2)
    labels_true=X[label]
    X=X.drop(['churn','appetency','upselling',label],axis='columns')
    db = est.fit(X,labels_true)
    randScore(db,labels_true)
    ###############################################################################
    # Visualize the results on PCA-reduced data
    data = scale(X)
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
    kmeans.fit(reduced_data)
    
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].
    
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
def auc_compute(actual,predictions):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()   
    # may differ
    print auc(false_positive_rate, true_positive_rate)
    print roc_auc_score(actual,predictions)
def gbc_features(X,y,columns_names):
    gbc = GradientBoostingClassifier()
    gbc.fit(X, y)
    # Get Feature Importance from the classifier
    feature_importance = gbc.feature_importances_
    # Normalize The Features
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    feature_importance= feature_importance[sorted_idx]
    feature_importance=[x for x in feature_importance if x >5]
    return np.asanyarray(columns_names)[sorted_idx][-len(feature_importance):]    
def split(X,y,test_size=0.1):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     X, y, test_size=test_size, random_state=0)
    return X_train, X_test, y_train, y_test 

def scores(X,y,y_proba,name="nan",to_plot=False):
#    print(name+' Classifier:\n {}'.format(metrics.classification_report(X,y)))
    cm= metrics.confusion_matrix(X,y)
    print cm
    if(to_plot):
        plt_cm(X,y,[-1,1])
        auc_compute(X,y)
    auc=roc_auc_score(X,y_proba)
    print(name+' Classifier auc:  %f' % auc)
    accuracy=metrics.accuracy_score(X,y)
    print(name+' Classifier accuracy:  %f' % (accuracy))
    f1=metrics.f1_score(X,y,pos_label=1)
    print(name+' Classifier f1: %f' % (f1))
    precision=metrics.precision_score(X,y)
    print(name+' Classifier precision_score: %f' % (precision))
    recall=metrics.recall_score(X,y)
    print(name+' Classifier recall_score: %f' % (recall))
    kappa_score=kappa(X,y)
    
    print(name+' Classifier kappa_score:%f' % (kappa_score))
    return [auc,f1.mean(),accuracy.mean(),precision.mean(),recall.mean(),kappa_score]
def standardize(X_train,X_test):
    scaler=preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    return X_train,X_test
    
    