# -*- coding: utf-8 -*-
"""
Created on Sun May 15 16:58:52 2016

@author: zero
"""

import Orange

lr = Orange.classification.logreg.LogRegLearner()
imputer = Orange.feature.imputation.MinimalConstructor

imlr = Orange.feature.imputation.ImputeLearner(base_learner=lr,
    imputer_constructor=imputer)

data = Orange.data.Table("voting")
for x in data.domain.features:
        n_miss = sum(1 for d in data if d[x].is_special())
        print "%4.1f%% %s" % (100.*n_miss/len(data), x.name)
```````````````