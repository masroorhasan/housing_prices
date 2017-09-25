# import data modules
import pandas as pd
import numpy as np
# import sklearn modules
from sklearn.metrics import r2_score, mean_squared_error

# build training and testing tests
training_set = pd.read_csv('data/train.csv')
testing_set = pd.read_csv('data/test.csv')

# print training set
# print training_set.head()
# TODO: clean up data by removing invalid entries

# split training and testing set to features and labels
# TODO: split training set further via cross validation?
training_set_features = training_set.drop('SalePrice', axis=1)
training_set_labels = training_set['SalePrice']
testing_set_features = testing_set['SalePrice']
testing_set_labels = testing_set.drop('SalePrice', axis=1)


'''
Calculate and print perf metrics of r2 score and RMSE
'''
def performance_metrics(y_true, y_predict):
    # calculate r2 score and print
    r2 = r2_score(y_true, y_predict)
    print "r2 score: {0:.2f}".format(r2)

    # calculate RMSE 
    rmse = mean_squared_error(y_true, y_predict)
    print "RMSE score: {0:.2f}".format(rmse)

