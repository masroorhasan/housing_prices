# import data modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
Explores categorical feature in data.
'''
def categorical_feature_exploration(data, feature):
    return data[feature].value_counts()

'''
Fills missing value of a feature in data.
'''
def fill_feature_missing_value(data, feature, value):
    data.loc[data[feature].isnull(),feature] = value
    return data

'''
Gets missing data columns.
'''
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data #.head(20)

'''
Gets missing features in data.
'''
def missing_features(data):
    return data.columns[data.isnull().any()].tolist()


'''
Prediction scatter plot.
'''
def pred_scatter_plot(actual, prediction):
    plt.figure(figsize=(10, 5))
    plt.scatter(actual, prediction, s=20)
    plt.title('Predicted vs. Actual')
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predicted Sale Price')
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)])
    plt.tight_layout()
    plt.show()

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

    '''
Data playground method with plot visualizations.
'''
def data_playground(training_set, testing_set):
    
    # few (relevant) features selected by eye balling the data
    test_features = ['OverallQual', 'OverallCond', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea']    # ~85%

    # heat map
    de.heat_map(training_set)

    # scatter plots of individual feature against saleprice
    relevant_features = ['GrLivArea', 'OverallQual']
    for feature_name in relevant_features:
        de.scatter_plot(training_set, feature_name)

    # scatter plot "large spot 1"
    large_spot_features_1 = ['TotalBsmtSF', '1stFlrSF']
    for feature_name in large_spot_features_1:
        de.scatter_plot(training_set, feature_name)

    # scatter plot "large spot 2"
    large_spot_features_2 = ['GarageYrBlt', 'GarageCars', 'GarageArea']
    for feature_name in large_spot_features_2:
        de.scatter_plot(training_set, feature_name)
    
    # features from heat map "spots"
    hm_features = relevant_features + large_spot_features_1 #+ large_spot_features_2 # ~84%

    # coorelation matrix
    de.correlation_matrix(training_set)
    largest_correlated_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt'] # ~82%
    de.scatter_pairplot(training_set, largest_correlated_features)
    
    # combine features from heatmap and correlated data exploration
    combined_features = largest_correlated_features + list(set(hm_features) - set(largest_correlated_features))
    # print combined_features

    # largest coorelation with sale price
    corr_mat = training_set.corr()
    corr_columns = corr_mat.nlargest(10, 'SalePrice')['SalePrice'].index
    corr_columns = filter(lambda x : x != 'SalePrice', corr_columns)
    # print "largest coorelation with SalePrice:"
    # print corr_columns
    # columns = corr_columns['Id']
    # training_set = training_set[columns]
    # testing_set = testing_set[columns]