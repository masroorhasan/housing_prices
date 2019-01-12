# import data modules
import pandas as pd
import numpy as np

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
