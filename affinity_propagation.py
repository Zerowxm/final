"""
=================================================
Demo of affinity propagation clustering algorithm
=================================================

Reference:
Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
Between Data Points", Science Feb. 2007

"""
print(__doc__)

import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn import metrics,datasets
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle
import utils 
##############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
                            random_state=0)
iris = datasets.load_iris()
X = iris.data
labels_true = iris.target
X,labels_true,target_names=utils.read_full()
estimators={}
estimators={'damping_0.5_convergence_iter_15_max_iter_200_preference_none':AffinityPropagation(),
            'damping_0.5_convergence_iter_15_max_iter_200_preference_-50':AffinityPropagation(preference=-50),
            'damping_0.5_convergence_iter_15_max_iter_400_preference_-50':AffinityPropagation(preference=-50,max_iter =400),
            'damping_1_convergence_iter_100_max_iter_1000_preference_-25':AffinityPropagation(damping=.9,max_iter =1000,convergence_iter =100,preference=-25),
            'damping_1_convergence_iter_30_max_iter_400_preference_-25':AffinityPropagation(damping=.9,max_iter =400,convergence_iter =30,preference=-25),
            'damping_0.5_convergence_iter_15_max_iter_200_preference_-50':AffinityPropagation(damping=.9,max_iter =400,convergence_iter =30,preference=-25),
            'damping_1_convergence_iter_30_max_iter_400_preference_none':AffinityPropagation(damping=.9,max_iter =400,convergence_iter =30,preference=None),
            'damping_0.5_convergence_iter_100_max_iter_1000_preference_-25':AffinityPropagation(damping=0.5,max_iter =1000,convergence_iter =100,preference=-25),
}

def ests_damping():
    for i in np.arange(0.5,1,0.05):
        print i
        estimators['damping'+str(i)]=AffinityPropagation(damping=i)
        
##############################################################################
# Compute Affinity Propagation
#ests_damping()
adjustD={}
clusters_n={}
for name, est in estimators.items():
    db = est.fit(X)
    labels = db.labels_
    adjustD[name]=metrics.adjusted_rand_score(labels_true,labels)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    clusters_n[name]=n_clusters_
    print('Estimated estimator: %s' % name)
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    plt.close('all')
    plt.figure(1)
    plt.clf()
    cluster_centers_indices = db.cluster_centers_indices_
    labels = db.labels_
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]
        plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        for x in X[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
    
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
      

##############################################################################
# Plot result
def plot_damp():
    G= sorted(adjustD.items(), key=lambda d: d[1]) 
    G.append(('average',np.mean([x[1] for x in G])))
    keys=[x[0] for x in G]
    values=[x[1] for x in G]
    adjustD['average']=np.mean(adjustD.values())
    plt.barh(np.arange(len(keys)),values,align='center',xerr=values)
    plt.yticks(range(len(keys)),keys)
    plt.title("Damping")
    plt.show()
    
    plt.plot(range(len(keys)),values)
    plt.show()
def plot_(mdict,c):
    mdict['average']=np.mean(mdict.values())
    G= sorted(mdict.items(), key=lambda d: d[1]) 
    keys=[x[0] for x in G]
    values=[x[1] for x in G]
    plt.barh(np.arange(len(keys)),values,align='center',xerr=values,color=c)
    plt.yticks(range(len(keys)),mdict.keys())
    plt.show()
    
    plt.plot(range(len(keys)),values)
    plt.show()
plot_(adjustD,'b')
#plot_damp()