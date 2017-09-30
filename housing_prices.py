# import python modules
import pandas as pd
import numpy as np
import datetime

# import modules
import data_exploration as de

# import sklearn modules
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.cross_validation import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

'''
Main method.
'''
def main():
    # build training and testing tests
    training_set = pd.read_csv('data/train.csv')
    testing_set = pd.read_csv('data/test.csv')
    # print training_set.head()
    # print testing_set.head()
    # print training_set.columns
    print training_set['SalePrice'].describe()

    # explore data with plots
    # data_playground(training_set, testing_set)

    # split training set to features and labels
    # training_labels = training_set['SalePrice']
    training_labels = training_set.pop('SalePrice')
    training_labels = np.log(training_labels)
    # training_set = training_set.drop('SalePrice', axis=1, inplace=True)
    features = pd.concat([training_set, testing_set], keys=['train', 'test'])
    # print features.loc['train']

    # drop missing features
    features = drop_missing_data(features)

    # transform feature data as needed
    features = transform_na_data(features)

    # get training features
    training_features = features.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

    # split data for train and test validation
    X_train, X_test, y_train, y_test = train_test_split(
        training_features, 
        training_labels,
        test_size=0.2, 
        random_state=42)
    
    # get decision tree regressor model
    dt_regressor_model = get_best_dt_regressor_model(X_train, y_train)

    # predict on validation sets
    pred = dt_regressor_model.predict(X_test)
    # perf_metric_rmse(y_test, pred)
    print "R2 score from dt_regressor_model pred: {0:.3f}".format(perf_metric_r2(y_test, pred))
    print "RMSE score from dt_regressor_model pred: {0:.3f}".format(perf_metric_rmse(y_test, pred))

    # get testing set from transformed datas
    testing_features = features.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

    # predict on testing set and save to csv
    pred_test_set = np.exp(dt_regressor_model.predict(testing_features))
    filename = "%s.csv" % (datetime.datetime.now().strftime("%Y-%m-%d,%H:%M:%S"))
    pd.DataFrame({'Id': testing_set.Id, 'SalePrice':pred_test_set}).to_csv(filename, index=False)

'''
Performs grid search cross validation over max depth parameter to get optimal DecisionTree regressor model
trained on input data [X, y].
'''
def get_best_dt_regressor_model(X, y):
    # create cross validation sets from training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.20, random_state=42)

    # create decision tree regressor object
    regressor = DecisionTreeRegressor()

    # set of parameters to test
    gs_params = {'max_depth':[None,2,5,10,20],
                'min_samples_split':[2,10,20],
                'min_samples_leaf':[1,5,10],
                'max_leaf_nodes':[None,5,10]
                }

    # use r2 as scoring function
    gs_scoring_func = make_scorer(perf_metric_r2)

    # create grid search object
    grid_search = GridSearchCV(regressor, param_grid=gs_params, scoring=gs_scoring_func, cv=cv_sets)

    # compute optimal model by fitting grid search object to data
    grid_search = grid_search.fit(X, y)
    model = grid_search.best_estimator_
    
    # print optimal params
    print "Optimal max_depth for model {}".format(model.get_params()['max_depth'])
    print "Optimal min_samples_split for model {}".format(model.get_params()['min_samples_split'])
    print "Optimal min_samples_leaf for model {}".format(model.get_params()['min_samples_leaf'])
    print "Optimal max_leaf_nodes for model {}".format(model.get_params()['max_leaf_nodes'])

    # return optimal model from fitted data
    return model

'''
Calculate and return r2 score.
'''
def perf_metric_r2(y_true, y_predict):
    # calculate r2 score and print
    r2 = r2_score(y_true, y_predict)
    # print "r2 score: {0:.3f}".format(r2)
    return r2

'''
Calculate and return RMSE.
'''
def perf_metric_rmse(y_true, y_predict):
    # calculate RMSE 
    rmse = np.sqrt(mean_squared_error(y_true, y_predict))
    # print "RMSE score: {0:.3f}".format(rmse)
    return rmse

'''
Drops missing data.
'''
def drop_missing_data(data):
    # missing data
    missing_data = de.missing_data(data)

    # print data with more than 1 null values
    print "data with > 1 null values:"
    print sorted((missing_data[missing_data['Total'] > 1]).index)
    
    # print data with 1 null value
    print "data with 1 null value:"
    print sorted((missing_data[missing_data['Total'] == 1]).index)
    # remove features missing over 15% of data
    # training_set = training_set.drop((missing_data[missing_data['Percent'] >= 0.15]).index, axis=1)
    updated_data = data.drop((missing_data[missing_data['Total'] > 1]).index, axis=1)
    # print training_set.isnull().sum().max()
    return updated_data

'''
'''
def transform_na_data(data):
    
    # fill NA with 0.0 for TotalBsmtSF
    # print "TotalBsmtSF missing:", data['TotalBsmtSF'].isnull().sum()
    data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(0.0)

    # fill NA with 0.0 for BsmtFinSF1
    data['BsmtFinSF1'] = data['BsmtFinSF1'].fillna(0.0)

    # fill NA with 0.0 for BsmtFinSF2
    data['BsmtFinSF2'] = data['BsmtFinSF2'].fillna(0.0)

    # fill NA with 0.0 for BsmtUnfSF
    data['BsmtUnfSF'] = data['BsmtUnfSF'].fillna(0.0)

    # fill NA with mode of set for Electrical
    # print "Electrical missing:", data['Electrical'].isnull().sum()
    data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])

    # fill NA with mode of set for Exterior1st
    data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])

    # fill NA with mode of set for Exterior2nd
    data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])

    # fill NA with mode of set for KitchenQual
    data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])

    # fill NA with 0 for GarageCars
    data['GarageCars'] = data['GarageCars'].fillna(0)

    # fill NA with 0 for GarageArea
    data['GarageArea'] = data['GarageArea'].fillna(0.0)

    # fill NA with mode of set for SaleType
    data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])

    # return transformed data
    return data

'''
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

'''
Entry point.
'''
if __name__ == '__main__':
    main()

