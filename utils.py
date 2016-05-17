# -*- coding: utf-8 -*-
"""
    Created on Sat Apr 16 18:14:53 2016

@author: Zero
"""
from random import  choice
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Imputer
from sklearn import tree
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc,roc_auc_score
import sklearn.linear_model as lm
from sklearn.neighbors import KNeighborsClassifier
from itertools import cycle
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 
from sklearn import svm   
from collections import Counter
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, make_scorer
plt.style.use('fivethirtyeight') # Good looking plots
pd.set_option('display.max_columns', None) # Display any number of columns

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

def GNBClassifier(X,label):
    gnb = GaussianNB()
    y=X[label]
    X=X.drop(['churn','appetency','upselling',label],axis='columns')
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     X, y, test_size=0.2, random_state=0)
    print Counter(y_train),Counter(y_test)
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"
      % (X.shape[0],(y_test != y_pred).sum()))

def knnClassifer(X,label):
    neigh = KNeighborsClassifier(n_neighbors=3)
    y=X[label]
    X=X.drop(['churn','appetency','upselling',label],axis='columns')
    neigh.fit(X,y)

def randScore(est,labels_true):
    labels = est.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))

from sklearn.feature_selection import RFE
def svmClassifer(X,label,sample_weight=None,cv=10):
    clfs={'NuSVC':svm.NuSVC(class_weight='balanced'),
          'LinearSVC':svm.LinearSVC()}
    y=X[label]
    X=X.drop(['churn','appetency','upselling',label],axis='columns')
    # Create the RFE object and rank each pixel
    svc = svm.SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
    rfe.fit(X, y)
#    ranking = rfe.ranking_.reshape(digits.images[0].shape)
    # Plot pixel ranking
#    plt.matshow(ranking)
#    plt.colorbar()
#    plt.title("Ranking of pixels with RFE")
#    plt.show()
    for name, clf in clfs.items():
        f1=cross(clf,X,y,name,cv=cv)
    return f1
def scores(X,y,name):
    print(name+' Classifier:\n {}'.format(metrics.classification_report(X,y)))
    cm= metrics.confusion_matrix(X,y)
    print cm
    class2=float(cm[1,1])/float(cm[1].sum())
    class1=float(cm[0,0])/float(cm[0].sum())
    print class1,class2
    accuracy=metrics.accuracy_score(X,y)
    print(name+' Classifier accuracy:  %0.2f (+/- %0.2f)' % (accuracy.mean(), accuracy.std()))
    f1=metrics.f1_score(X,y)
    print(name+' Classifier f1: %0.2f (+/- %0.2f)' % (f1.mean(), f1.std()))
    precision=metrics.precision_score(X,y)
    print(name+' Classifier precision_score: %0.2f (+/- %0.2f)' % (precision.mean(), precision.std()))
    recall=metrics.recall_score(X,y)
    print(name+' Classifier recall_score: %0.2f (+/- %0.2f)' % (recall.mean(), recall.std()))
    
    
    return [f1.mean(),accuracy.mean(),precision.mean(),recall.mean(),class1,class2]
    
def cross(est,X,y,name,cm=False,cv=10):
    
    kappa_scorer = make_scorer(cohen_kappa_score)
    if(cm):
        print(metrics.classification_report(expected,predicted))  
        print(metrics.confusion_matrix(expected,predicted))
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
    
def cutoff_predict(clf,X,cutoff):
    return (clf.predict_proba(X)[:,1]>cutoff).astype(int)

def custom_f1(cutoff):
    def f1_cutoff(clf,X,y):
        ypred= cutoff_predict(clf,X,cutoff)
        return metrics.f1_score(y,ypred)
    return f1_cutoff
    
def ploy_standard_scale(df):
    X = df.as_matrix().astype(np.float)
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    polynomial_features = preprocessing.PolynomialFeatures()
    X = polynomial_features.fit_transform(X)
    return X
    
def test(X,label, sample_weight=None,cv=10):
    scores=[]
    y=X[label]
    X=X.drop(['churn','appetency','upselling',label],axis='columns')
    for cutoff in np.arange(0.1,0.9,0.1):
        clf=RandomForestClassifier(n_estimators=15) 
        valideted=cross_validation.cross_val_score(clf,X,y,scoring=custom_f1(cutoff))
        print(" f1: %0.2f (+/- %0.2f)" % (valideted.mean(), valideted.std()))
        scores.append(valideted)
def selectF(X,label):
    est = DecisionTreeClassifier(min_samples_split=1)
    y=X[label]
    X=X.drop(['churn','appetency','upselling',label],axis='columns')
    is_poly=True
    if(is_poly):
        X = X.as_matrix().astype(np.float)
        polynomial_features = preprocessing.PolynomialFeatures()
        X = polynomial_features.fit_transform(X)
    else:
        X=X.as_matrix().astype(np.float)
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
    y=y.as_matrix().astype(np.int)
    f1=scores(y,stratified_cv(X, y, DecisionTreeClassifier),'DT')
    return f1
def treeClassifer(X,label, sample_weight=None,cv=10):
    est = DecisionTreeClassifier(min_samples_split=1)
    y=X[label]
    X=X.drop(['churn','appetency','upselling',label],axis='columns')
    is_poly=True
    if(is_poly):
        X = X.as_matrix().astype(np.float)
        polynomial_features = preprocessing.PolynomialFeatures()
        X = polynomial_features.fit_transform(X)
    else:
        X=X.as_matrix().astype(np.float)
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
#    est = est.fit(X, y,sample_weight=sample_weight)
    f1=cross(est,X,y,'DT',cv=cv)
    y=y.as_matrix().astype(np.int)
#    scores(y,stratified_cv(X, y, DecisionTreeClassifier),'DT')
    
    return f1
      
def predict(X,label,est,name):
    y=X[label]
    X=X.drop(['churn','appetency','upselling',label],axis='columns')
    X=preprocessing.scale(X)
    est = est.fit(X, y)
    expected=y
    predicted=est.predict(X)
    print(metrics.classification_report(expected,predicted))  
    print(metrics.confusion_matrix(expected,predicted))
    f1=cross(est,X,y,name)
    return f1
from sklearn import linear_model,neighbors,ensemble

def cof(X,y):
    pass_agg_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, linear_model.PassiveAggressiveClassifier))
    grad_ens_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))
    decision_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, tree.DecisionTreeClassifier))
    ridge_clf_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, linear_model.RidgeClassifier))
    svm_svc_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, svm.SVC))
    random_forest_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, ensemble.RandomForestClassifier))
    k_neighbors_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))
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
        
def gbc(X,y,columns_names):
    gbc = ensemble.GradientBoostingClassifier()
    gbc.fit(X, y)
    # Get Feature Importance from the classifier
    feature_importance = gbc.feature_importances_
    # Normalize The Features
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(16, 12))
    plt.barh(pos, feature_importance[sorted_idx], align='center', color='#7A68A6')
    plt.yticks(pos, np.asanyarray(columns_names)[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()    

def classification(X,label):
    y=X[label]
    X=X.drop(['churn','appetency','upselling',label],axis='columns')
    clf=ExtraTreesClassifier(n_estimators=10,max_depth=None,min_samples_split=1,random_state=0)
    clf1 = LogisticRegression()
    clf2 = RandomForestClassifier()
    clf3 = GaussianNB()
    clf4 = AdaBoostClassifier(n_estimators=100)
    clf5 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    clf6 = DecisionTreeClassifier(max_depth=None, min_samples_split=1,random_state=0)
    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), 
                                        ('gnb', clf3),('ab',clf4),('gd',clf5),('dt',clf6),('ETs',clf)],
                                         voting='soft')
    eclf.fit(X, y)
    
def convert(x):
     le = preprocessing.LabelEncoder()
     new_x=le.fit_transform(x)
     return new_x
     
def inpute(X):
    c=X.columns
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    return pd.DataFrame(imp.fit_transform(X),columns=c)   
def selectFeaturesThres(X):
    c=X.columns
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    return pd.DataFrame(sel.fit_transform(X),columns=c)
 
def selectFeatures(X,y,k):
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(X, y)
    X_new = SelectKBest(f_classif, k=k).fit_transform(X, y)
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
  
def bench_k_means(estimator, name, data,labels):
    print('% 9s' % 'init     '
      '    time  inertia    homo   compl  v-meas     ARI AMI')
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_)))

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
def plot_rfe(X,label):
    y=X[label]
    X=X.drop(['churn','appetency','upselling',label],axis='columns')
    from sklearn.svm import SVC
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.feature_selection import RFECV
    import sklearn.linear_model as lm
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
    cross(forest,X,y,'ETs')
    return np.array(indices)[:20]

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

def SMOTE(T, N, k):
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
