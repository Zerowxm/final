"""Kaggle competition: Predicting a Biological Response.

Blending {RandomForests, ExtraTrees, GradientBoosting} + stretching to
[0,1]. The blending scheme is related to the idea Jose H. Solorzano
presented here:
http://www.kaggle.com/c/bioresponse/forums/t/1889/question-about-the-process-of-ensemble-learning/10950#post10950
'''You can try this: In one of the 5 folds, train the models, then use
the results of the models as 'variables' in logistic regression over
the validation data of that fold'''. Or at least this is the
implementation of my understanding of that idea :-)

The predictions are saved in test.csv. The code below created my best
submission to the competition:
- public score (25%): 0.43464
- private score (75%): 0.37751
- final rank on the private leaderboard: 17th over 711 teams :-)

Note: if you increase the number of estimators of the classifiers,
e.g. n_estimators=1000, you get a better score/rank on the private
test set.

Copyright 2012, Emanuele Olivetti.
BSD license, 3 clauses.
"""

from __future__ import division
import numpy as np
import load_data
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,VotingClassifier
from sklearn.tree import DecisionTreeClassifier
def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
    return - np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt))

import utils as u
if __name__ == '__main__':

    np.random.seed(0) # seed to shuffle the train set

    n_folds = 10
    verbose = True
    shuffle = False
#    X_train, X_test, y_train, y_test=u.split(X_,y_,test_size=0.1)
#    X, y, X_submission = load_data.load()
#    X=X_train.as_matrix()
#    y=y_train.as_matrix()
#    X_submission=X_test.as_matrix()
    y_test.to_csv('y_test1.csv')
    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))
    rf=RandomForestClassifier(n_estimators=100,class_weight ='balanced') 
    gbc = GradientBoostingClassifier(n_estimators=50,learning_rate=0.05).fit(X_train,y_train)
    ets=ExtraTreesClassifier(n_estimators=100,max_depth=None,min_samples_split=1,random_state=0)
    ab = AdaBoostClassifier()
    clfs = [
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            AdaBoostClassifier(n_estimators=100, base_estimator=DecisionTreeClassifier(),learning_rate=1),
            GradientBoostingClassifier( subsample=0.5, max_depth=7, n_estimators=50)]
#,
#            VotingClassifier(estimators=[ ('rf', rf), 
#                                        ('gd',gbc),('ETs',ets),('ab',ab)],
#                                         voting='soft',weights =[2,3,1,2])
    print "Creating train and test sets for blending."
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    
    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
           
            X_trai = X[train]
            y_trai = y[train]
            X_tes = X[test]
            y_tes = y[test]
            X_trai,y_trai=u.test_rest(X_trai,y_trai,ratio=4)
#            X_trai,y_trai=u.test_smote(X_trai,y_trai,c=0)
#            if(i==(len(clfs)-1)):
#                rf.fit(X_trai, y_trai)
#                gbc.fit(X_trai, y_trai)
#                ets.fit(X_trai, y_trai)
#                ab.fit(X_trai, y_trai)
            clf.fit(X_trai, y_trai)
            y_submission = clf.predict_proba(X_tes)[:,1]
#            print u.roc_auc_score(y_tes,y_submission),u.scores(y_tes,clf.predict(X_tes))
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    print "Blending."
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    print "Linear stretch of predictions to [0,1]"
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    y_test[y_test==1].index
    u.roc_auc_score(y_test,y_submission,average='micro', sample_weight=None)
    u.roc_auc_score(y_test,y_submission,average='macro', sample_weight=None)
    u.roc_auc_score(y_test,y_submission,average='weighted', sample_weight=None)
    u.roc_auc_score(y_test,y_submission,average='samples', sample_weight=None)
    print "Saving Results."
    np.savetxt(fname='y_submission1.csv', X=y_submission, fmt='%0.9f')
    
