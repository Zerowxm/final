# -*- coding: utf-8 -*-
"""
    Created on Sat Apr 16 18:14:53 2016

@author: Zero
"""

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
     X, y, test_size=0.4, random_state=0)
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
def km(X,label):
    from sklearn.cluster import KMeans
    est=KMeans(n_clusters=2)
    labels_true=X[label]
    X=X.drop(['churn','appetency','upselling',label],axis='columns')
    db = est.fit(X,labels_true)
    randScore(db,labels_true)
#    random_state = 170
#    y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)
#    
#    plt.subplot(221)
#    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
#    plt.title("Incorrect Number of Blobs")
    
    
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

def LRClassifer(X,label):
    clf = lm.LogisticRegression()
    y=X[label]
    X=X.drop(['churn','appetency','upselling',label],axis='columns')
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     X, y, test_size=0.2, random_state=0)
    clf = clf.fit(X_train, y_train)
    print(label+":\n%s" % (
        metrics.classification_report(
        y_test,
        clf.predict(X_test))))
    print roc_auc_score(y_test, clf.predict(X_train))
    plt_cm(y_test,clf.predict(X_test),[-1,1])
def cross(est,X,y,name):
    print('10-fold cross validation:\n')
    start_time = time()
    scores = cross_validation.cross_val_score(est, X,y, cv=10, scoring ='f1')
    print(" f1: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), name))
    print("---Classifier %s use %s seconds ---" %(name, (time() - start_time)))
    
def featureImp(X,y,forest):
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
#    print("Feature ranking:")
#    
#    for f in range(X.shape[1]):
#        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#    
#    # Plot the feature importances of the forest
#    plt.figure()
#    plt.title("Feature importances")
#    plt.bar(range(X.shape[1]), importances[indices],
#           color="r", yerr=std[indices], align="center")
#    plt.xticks(range(X.shape[1]), indices)
#    plt.xlim([-1, X.shape[1]])
#    plt.show()
    cross(forest,X,y,'ETs')
    return np.array(indices)[:20]
    
def treeClassifer(X,label):
    est = DecisionTreeClassifier()
    y=X[label]
    columns=X.columns
    X=X.drop(['churn','appetency','upselling',label],axis='columns')
    X=preprocessing.scale(X)
#    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
#     X, y, test_size=0.4, random_state=0)
#    clf = clf.fit(X_train, y_train)
    est = est.fit(X, y)
    cross(est,X,y,'DT')
    
    importances = est.feature_importances_
#    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
#                 axis=0)
    indices = np.argsort(importances)[::-1]
    features=columns[indices]
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r",  align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    print indices[:20]
    return indices[:20]
#    best_features= intersection(featureImp(X,y,ExtraTreesClassifier(n_estimators=250,
#                              random_state=0))[:10],indices[:10])
#    print best_features
      
                    
#    print(label+":\n%s" % (
#        metrics.classification_report(
#        y_test,
#        clf.predict(X_test))))
#    plt_cm(y_test,clf.predict(X_test),[-1,1])
    
def intersection(a,b):
    print a,b
    return list(set(a).intersection(set(b)))

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
     return le.fit_transform(x)
     
def inpute(X):
    c=X.columns
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    return pd.DataFrame(imp.fit_transform(X),columns=c)   
      
def inpute_category(X):
    c=X.columns
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    return pd.DataFrame(imp.fit_transform(X),columns=c)
    
def normalize_df(X):
    c=X.columns
    # normalize the data attributes
    normalized_X = preprocessing.normalize(X)
# standardize the data attributes
    standardized_X = preprocessing.scale(X)
    
    min_max_X=preprocessing.MinMaxScaler(X)
    
    return pd.DataFrame(min_max_X,columns=c)
   
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
  
def selectFeaturesThres(X):
    c=X.columns
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    return pd.DataFrame(sel.fit_transform(X),columns=c)
 
@fn_timer 
def selectFeatures(X,y,k):
#    sel = VarianceThreshold(threshold=())
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
#    plt.close('all')
#    plt.figure(1)
#    plt.clf()
#    cluster_centers_indices = db.cluster_centers_indices_
#    labels = db.labels_
#    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
#    for k, col in zip(range(n_clusters_), colors):
#        class_members = labels == k
#        cluster_center = X[cluster_centers_indices[k]]
#        plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
#        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#                 markeredgecolor='k', markersize=14)
#        for x in X[class_members]:
#            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
#    
#    plt.title('Estimated number of clusters: %d' % n_clusters_)
#    plt.show()
             
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
    
def k_fold_generator(X, y, k_fold):
    subset_size = len(X) / k_fold  # Cast to int if using Python 3
    for k in range(k_fold):
        X_train = X[:k * subset_size] + X[(k + 1) * subset_size:]
        X_valid = X[k * subset_size:][:subset_size]
        y_train = y[:k * subset_size] + y[(k + 1) * subset_size:]
        y_valid = y[k * subset_size:][:subset_size]

        yield X_train, y_train, X_valid, y_valid
#    import nltk
#    from sklearn import cross_validation
#    training_set = nltk.classify.apply_features(extract_features, documents)
#    cv = cross_validation.KFold(len(training_set), n_folds=10, indices=True, shuffle=False, random_state=None, k=None)
#    
#    for traincv, testcv in cv:
#        classifier = nltk.NaiveBayesClassifier.train(training_set[traincv[0]:traincv[len(traincv)-1]])
#        print 'accuracy:', nltk.classify.util.accuracy(classifier, training_set[testcv[0]:testcv[len(testcv)-1]])