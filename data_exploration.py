# import data modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
Scatter plot.
'''
def scatter_plot(data, feature, label='SalePrice'):
    xy = pd.concat([data[label], data[feature]], axis=1)
    xy.plot.scatter(x=feature, y=label, ylim=(0,800000));
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