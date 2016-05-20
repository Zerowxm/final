# -*- coding: utf-8 -*-
"""
Created on Wed May 11 18:10:40 2016

@author: zero
"""

import Orange
bridges = Orange.data.Table("bridges")
import ut
imputer = Orange.feature.imputation.MinimalConstructor()
imputer = imputer(bridges)
data =bridges[10]
print "Example with missing values"
print data
print "Imputed values:"
print imputer(data)

imputed_bridges = imputer(bridges)
print imputed_bridges[10]

imputer = Orange.feature.imputation.ModelConstructor()
imputer.learner_continuous = imputer.learner_discrete = Orange.classification.tree.TreeLearner(min_subset=20)
imputer = imputer(bridges)
print imputer(data)
print bridges.domain.features
#imputer = imputer([1,1,1,1])
print bridges.has_missing_values()
a=imputed_bridges.to_numpy()[0]
#t= ut.table2df(imputed_bridges)