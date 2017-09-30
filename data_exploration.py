# import data modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
Get missing data columns.
'''
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data #.head(20)

'''
Scatter plot.
'''
def scatter_plot(data, feature, label='SalePrice'):
    xy = pd.concat([data[label], data[feature]], axis=1)
    xy.plot.scatter(x=feature, y=label, ylim=(0,800000));
    plt.show()

'''
Scatter pair plot.
'''
def scatter_pairplot(data, features):
    sns.set()
    sns.pairplot(data[features], size = 2.5)
    plt.show()

'''
Histogram.
'''
def histogram(data):
    sns.distplot(data)
    plt.show()

'''
Heat map.
'''
def heat_map(data):
    corr_mat = data.corr()   # correlation matrix
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr_mat, vmax=.8, square=True)
    plt.show()

'''
Correlation matrix.
'''
def correlation_matrix(data, n=10):
    # n largest from correlation matrix
    corr_mat = data.corr()
    cols = corr_mat.nlargest(n, 'SalePrice')['SalePrice'].index
    print cols
    
    # draw heat
    hm_data = np.corrcoef(data[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(hm_data, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()