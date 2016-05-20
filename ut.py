# -*- coding: utf-8 -*-
"""
Created on Sat May 14 10:52:29 2016

@author: zero
"""

import pandas as pd
import numpy as np
import Orange

#### For those who are familiar with pandas
#### Correspondence:
####    value <-> Orange.data.Value
####        NaN <-> ["?", "~", "."] # Don't know, Don't care, Other
####    dtype <-> Orange.feature.Descriptor
####        category, int <-> Orange.feature.Discrete # category: > pandas 0.15
####        int, float <-> Orange.feature.Continuous # Continuous = core.FloatVariable
####                                                 # refer to feature/__init__.py
####        str <-> Orange.feature.String
####        object <-> Orange.feature.Python
####    DataFrame.dtypes <-> Orange.data.Domain
####    DataFrame.DataFrame <-> Orange.data.Table = Orange.orange.ExampleTable 
####                              # You will need this if you are reading sources

def series2descriptor(d, discrete=False):
    if d.dtype is np.dtype("float"):
        return Orange.feature.Continuous(str(d.name))
    elif d.dtype is np.dtype("int"):
        return Orange.feature.Continuous(str(d.name), number_of_decimals=0)
    else:
        t = d.unique()
        if discrete or len(t) < len(d) / 2:
            t.sort()
            return Orange.feature.Discrete(str(d.name), values=list(t.astype("str")))
        else:
            return Orange.feature.String(str(d.name))
def construct_domain(df):
    from collections import OrderedDict
    columns = OrderedDict(df.dtypes)

    def create_variable(col):
        if col[1].__str__().startswith('float'):
            return Orange.data.ContinuousVariable(col[0])
        if col[1].__str__().startswith('int') and len(df[col[0]].unique()) > 50:
            return Orange.data.ContinuousVariable(col[0])
        if col[1].__str__().startswith('date'):
            df[col[0]] = df[col[0]].values.astype(np.str)
        if col[1].__str__() == 'object':
            df[col[0]] = df[col[0]].astype(type(""))

        return Orange.data.DiscreteVariable(col[0], values = df[col[0]].unique().tolist())

    return Orange.data.Domain(list(map(create_variable, columns.items())))
def pandas_to_orange(df):
    domain = construct_domain(df)
    orange_table = Orange.data.Table.from_list(domain = domain, rows = df.values.tolist())
    return orange_table

def df2domain(df):
    featurelist = [series2descriptor(df.icol(col)) for col in xrange(len(df.columns))]
    return Orange.data.Domain(featurelist)
def convert_dataframe_to_orange(df):
    import os
    df.to_csv('_temp_.csv',index_label='id')
    orange_table = Orange.data.Table('_temp_.csv')
    os.unlink('_temp_.csv')
    return orange_table

def df2table(df):
    # It seems they are using native python object/lists internally for Orange.data types (?)
    # And I didn't find a constructor suitable for pandas.DataFrame since it may carry
    # multiple dtypes
    #  --> the best approximate is Orange.data.Table.__init__(domain, numpy.ndarray),
    #  --> but the dtype of numpy array can only be "int" and "float"
    #  -->  * refer to src/orange/lib_kernel.cpp 3059:
    #  -->  *    if (((*vi)->varType != TValue::INTVAR) && ((*vi)->varType != TValue::FLOATVAR))
    #  --> Documents never mentioned >_<
    # So we use numpy constructor for those int/float columns, python list constructor for other

    tdomain = df2domain(df)
    ttables = [series2table(df.icol(i), tdomain[i]) for i in xrange(len(df.columns))]
    return Orange.data.Table(ttables)

    # For performance concerns, here are my results
    # dtndarray = np.random.rand(100000, 100)
    # dtlist = list(dtndarray)
    # tdomain = Orange.data.Domain([Orange.feature.Continuous("var" + str(i)) for i in xrange(100)])
    # tinsts = [Orange.data.Instance(tdomain, list(dtlist[i]) )for i in xrange(len(dtlist))] 
    # t = Orange.data.Table(tdomain, tinsts)
    #
    # timeit list(dtndarray)  # 45.6ms
    # timeit [Orange.data.Instance(tdomain, list(dtlist[i])) for i in xrange(len(dtlist))] # 3.28s
    # timeit Orange.data.Table(tdomain, tinsts) # 280ms

    # timeit Orange.data.Table(tdomain, dtndarray) # 380ms
    #
    # As illustrated above, utilizing constructor with ndarray can greatly improve performance
    # So one may conceive better converter based on these results


def series2table(series, variable):
    if series.dtype is np.dtype("int") or series.dtype is np.dtype("float"):
        # Use numpy
        # Table._init__(Domain, numpy.ndarray)
        return Orange.data.Table(Orange.data.Domain(variable), series.values[:, np.newaxis])
    else:
        # Build instance list
        # Table.__init__(Domain, list_of_instances)
        tdomain = Orange.data.Domain(variable)
        tinsts = [Orange.data.Instance(tdomain, [i]) for i in series]
        return Orange.data.Table(tdomain, tinsts)
        # 5x performance


def column2df(col):
    if type(col.domain[0]) is Orange.feature.Continuous:
        return (col.domain[0].name, pd.Series(col.to_numpy()[0].flatten()))
    else:
        tmp = pd.Series(np.array(list(col)).flatten())  # type(tmp) -> np.array( dtype=list (Orange.data.Value) )
        tmp = tmp.apply(lambda x: str(x[0]))
        return (col.domain[0].name, tmp)

def table2df(tab):
    # Orange.data.Table().to_numpy() cannot handle strings
    # So we must build the array column by column,
    # When it comes to strings, python list is used
    series = [column2df(tab.select(i)) for i in xrange(len(tab.domain))]
    series_name = [i[0] for i in series]  # To keep the order of variables unchanged
    series_data = dict(series)
    print series_data
    return pd.DataFrame(series_data, columns=series_name)
    
#df2 = pd.DataFrame(np.random.randn(10, 5))
#a=df2.as_matrix()
##df2.columns=['churn','b']
#df2table=df2table(df2)
#
#bridges = Orange.data.Table("bridges")
#df3 = table2df( df2table )
#d = Orange.data.Domain([Orange.feature.Continuous('a%i' % x) for x in range(5)])
    
d = [['a', 'b'], ['e', 'f', 'g']]
vars = {}
def var_construct(name):
    if name not in vars:
        v = Orange.feature.Continuous(name)
        mid = Orange.feature.Descriptor.new_meta_id()
        domain.add_meta(mid, v, 1)
        vars[name] = v
    return vars[name]

#domain = Orange.data.Domain([])
#table = Orange.data.Table(domain)
#
#for inst in d:
#    inst_vars = map(var_construct, inst)
#    new_instance = Orange.data.Instance(domain)
#    for v in inst_vars:
#        new_instance[v] = 1.0
#    table.append(new_instance)
#table=table2df(table)
#a = np.array([[ 1,  0.2,  np.nan],
#       [ np.nan,  np.nan,  0.5],
#       [ np.nan,  0.2,  0.5],
#       [ 0.1,  0.2,  np.nan],
#       [ 0.1,  0.2,  0.5],
#       [ 0.1,  np.nan,  1],
#       [ 0.1,  np.nan,  np.nan]])
#data = Orange.data.Table(a)
#print data.domain.features
#imputer = Orange.feature.imputation.MinimalConstructor()
##imputer = imputer(data)
#data = Orange.data.Table("bridges")
#print data.domain.features
#for x in data.domain.features:
#    n_miss = sum(1 for d in data if d[x].is_special())
#    print "%4.1f%% %s" % (100.*n_miss/len(data), x.name)

#t= table2df(t)import Orange

#cards = [3, 3, 2, 3, 4, 2]
#values = ["1", "2", "3", "4",'d']
#
#features = [Orange.feature.Discrete(name, values=values)
#              for name, card in zip("abcdef", cards)]
#con=Orange.feature.Continuous('g')
#classattr = Orange.feature.Discrete("y", values=["0", "1"])
#print features,classattr
#domain = Orange.data.Domain(features +[con]+ [classattr])
#loe = [["d", "1", "1", "1", "?", "1",'?', "1"],
#       ["3", "1", "1", "2","1", "1", '?',"0"],
#       ["3", "3", "1", "2", "2", "1",2 ,"1"]
#      ]
#
#data = Orange.data.Table(domain,loe)
##print table2df(data),data.domain.class_var
#for x in data.domain.features:
#    n_miss = sum(1 for d in data if d[x].is_special())
#    print "%4.1f%% %s" % (100.*n_miss/len(data), x.name)
###df= table2df(data)
#d = Orange.data.Domain([Orange.feature.Continuous('b%i' % x) for x in range(5)])
#a = np.array([[1.1, 2, np.nan, 4, 5], [5, 4, 3, 2, 1]])
#data = Orange.data.Table(d,a)
#for x in data.domain.features:
#    n_miss = sum(1 for d in data if d[x].is_special())
#    print "%4.1f%% %s" % (100.*n_miss/len(data), x.name)
#t= table2df(data)
