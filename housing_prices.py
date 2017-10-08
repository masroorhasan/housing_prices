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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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
    # print training_set.columns
    print training_set['SalePrice'].describe()

    # fill training data values
    print "Missing data in TRAIN features"
    a = sorted(de.missing_features(training_set))
    print a

    print "Missing data in TEST features"
    b = sorted(de.missing_features(testing_set))
    print b

    print "Intersect:"
    print sorted(list(set(b) & set(a)))

    print "Difference:"
    print sorted(list(set(b) - set(a)))

    print "Before filling in missing TRAIN data features:"
    print training_set[de.missing_features(training_set)].isnull().sum()
    training_set = transform_missing_feature_data(training_set)
    training_set = transform_special_train_data(training_set)
    print "After filling in missing TRAIN data features"
    print training_set[de.missing_features(training_set)].isnull().sum()
    
    print "Before filling in missing TEST data features:"
    print testing_set[de.missing_features(testing_set)].isnull().sum()
    testing_set = transform_missing_feature_data(testing_set)
    testing_set = transform_special_test_data(testing_set)
    print "After filling in missing TEST data features"
    print testing_set[de.missing_features(testing_set)].isnull().sum()

    # split training set to features and labels
    training_labels = training_set.pop('SalePrice')
    training_labels = np.log(training_labels)   # normalize with log transform
    features = pd.concat([training_set, testing_set], keys=['train', 'test'])
    # print features.loc['train']
    # return

    # drop missing features
    # features = drop_missing_data(features)
    
    # transform feature data as needed
    # features = transform_missing_feature_values(features)
    
    # get training features
    training_features = features.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

    # split data for train and test cross validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        training_features, 
        training_labels,
        test_size=0.2, 
        random_state=42)
    
    ##
    ## GradientBoosting regressor ##
    # Get model with best params fitted on cross validation set
    # gb_regressor_model = get_best_gb_regressor_model(X_train, y_train)
    # gb_params = gb_regressor_model.best_params_
    gb_params = {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 3, 'min_samples_leaf': 10, 'min_samples_split': 5}
    gb_regressor_model = GradientBoostingRegressor(random_state=42, **gb_params)
    gb_regressor_model = gb_regressor_model.fit(X_train, y_train)
    print_model_prediction_scores(gb_regressor_model, X_test, y_test)
    # create new gradientboosting regressor with params and train against full set
    best_gb_regressor = GradientBoostingRegressor(random_state=42, **gb_params)
    best_gb_regressor.fit(training_features, training_labels)

    ##
    ## RandomForest regressor ##
    # Get model with best params fitted on cross validation set
    # rf_regressor_model = get_best_rf_regressor_model(X_train, y_train)
    # rf_params = rf_regressor_model.best_params_
    rf_params = {'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 5, 'min_samples_split': 10}
    rf_regressor_model = RandomForestRegressor(random_state=42, n_jobs=5, **rf_params)
    rf_regressor_model = rf_regressor_model.fit(X_train, y_train)
    print_model_prediction_scores(rf_regressor_model, X_test, y_test)
    # create new randomforest regressor with params and train against full set
    best_rf_regressor = RandomForestRegressor(random_state=42, n_jobs=5, **rf_params)#**rf_regressor_model.best_params_)
    best_rf_regressor.fit(training_features, training_labels)

    ##
    ## Ensemble and predict ##
    # predict on testing set and save to csv
    testing_features = features.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
    pred_test_set = (
        # np.exp(dt_regressor_model.predict(testing_features)) +
        np.exp(best_gb_regressor.predict(testing_features)) +
        np.exp(best_rf_regressor.predict(testing_features))
        ) / 2
    filename = "%s.csv" % (datetime.datetime.now().strftime("%Y-%m-%d,%H:%M:%S"))
    pd.DataFrame({'Id': testing_set.Id, 'SalePrice':pred_test_set}).to_csv(filename, index=False)

'''
Prints model prediction scores based on [X, y] data.
'''
def print_model_prediction_scores(model, X, y):
    # print model details
    print(model)

    # precit from set
    predict = model.predict(X)
    print "prediction score of cv set:"
    print "R2 score: {0:.3f}".format(perf_metric_r2(y, predict))
    print "RMSE score: {0:.3f}".format(perf_metric_rmse(y, predict))
    # de.pred_scatter_plot(y, predict)
    print "\n"

'''
Gets best GradientBoost regressor model based on grid search, trained on data [X, y].
'''
def get_best_gb_regressor_model(X, y):
    # create cross validation sets from training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.20, random_state=42)

    # create the gradient boost regressor object
    regressor = GradientBoostingRegressor(random_state=42)
    
    # params to tune
    gs_params = {
        'n_estimators':[500,1000],
        'learning_rate':[0.05,0.1],
        'max_depth':[3,5],
        'min_samples_leaf':[5,10],
        'min_samples_split':[2,5],
        # 'min_samples_leaf':[15,30],
        # 'subsample'[1.0,2.0]
        # 'max_leaf_nodes':[None,5]
    }

    # use r2 as scoring function
    gs_scoring_func = make_scorer(perf_metric_r2)

    # create grid search object
    grid_search = GridSearchCV(regressor, param_grid=gs_params, scoring=gs_scoring_func, cv=cv_sets)

    # compute optimal model by fitting grid search object to data
    model = grid_search.fit(X, y)
    # model = grid_search.best_estimator_

    # print optimal params
    print "GradientBoosting"
    print model.best_params_

    # return the model
    return model

'''
Gets best RandomForest regressor model based on grid search, trained on data [X, y].
'''
def get_best_rf_regressor_model(X, y):
    # create cross validation sets from training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.20, random_state=42)

    # create the gradient boost regressor object
    regressor = RandomForestRegressor(n_jobs=5, random_state=42)
    
    # params to tune
    gs_params = {
        'n_estimators':[100,150,200],
        'max_depth':[5,10],
        'min_samples_leaf':[2,5],
        'min_samples_split':[5,10],
    }

    # use r2 as scoring function
    gs_scoring_func = make_scorer(perf_metric_r2)

    # create grid search object
    grid_search = GridSearchCV(regressor, param_grid=gs_params, scoring=gs_scoring_func, cv=cv_sets)

    # compute optimal model by fitting grid search object to data
    model = grid_search.fit(X, y)
    print "RandomForest"
    print model.best_params_

    # return the model
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
    # drop features missing more 1 data point
    updated_data = data.drop((missing_data[missing_data['Total'] > 1]).index, axis=1)
    # print training_set.isnull().sum().max()
    return updated_data

'''
Transforms missing feature values in both training and testing data set.
'''
def transform_missing_feature_data(data):
    
    # Fill NA with 'None' for categorical missing features
    data = de.fill_feature_missing_value(data, 'Alley', 'None')

    # Bsmt* missing features
    bsmt_features = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']
    # Missing basement features have same indices where total basement square footage is 0.0 (i.e. no basement)
    
    # print data['TotalBsmtSF'][data['TotalBsmtSF'] == 0.0].index
    # print data[bsmt_features][data['BsmtQual'].isnull()==True].index
    # print data['TotalBsmtSF'][data['TotalBsmtSF'] == 0.0].index
    print np.array_equal(data[bsmt_features][data['BsmtCond'].isnull()==True].index, data['TotalBsmtSF'][data['TotalBsmtSF'] == 0.0].index)
    for bmst_feature in bsmt_features:
        data = de.fill_feature_missing_value(data, bmst_feature, 'None')

    # Fill missing Fence, FireplaceQu with None
    data = de.fill_feature_missing_value(data, 'Fence', 'None')
    data = de.fill_feature_missing_value(data, 'FireplaceQu', 'None')

    # Garage* missing features
    garage_features = ['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt']
    print np.array_equal(data[garage_features][data['GarageCond'].isnull()==True].index, data['GarageArea'][data['GarageArea'] == 0.0].index)
    # print 'float64'==data['GarageYrBlt'].dtype
    # Fill missing features accordingly
    for garage_feature in garage_features:
        if 'float64'==data[garage_feature].dtype:
            data = de.fill_feature_missing_value(data, garage_feature, 0)
        else:
            data = de.fill_feature_missing_value(data, garage_feature, 'None')

    # LotFrontage: Linear feet of street connected to property
    # Check coorelations between relevant features
    # print data['LotFrontage'].corr(data['LotArea'])
    # print data['LotFrontage'].corr(np.sqrt(data['LotArea']))
    # Fill LotFrontage with mean for now
    # TODO: Update
    data = de.fill_feature_missing_value(data, 'LotFrontage', data['LotFrontage'].mean())

    # Masonry missing features
    # Missing values on same indices
    print np.array_equal(data[data['MasVnrArea'].isnull()==True].index, data[data['MasVnrType'].isnull()==True].index)
    data = de.fill_feature_missing_value(data, 'MasVnrArea', 0.0)
    data = de.fill_feature_missing_value(data, 'MasVnrType', 'None')

    # Fill MiscFeature, PoolQC with None values
    data = de.fill_feature_missing_value(data, 'MiscFeature', 'None')
    data = de.fill_feature_missing_value(data, 'PoolQC', 'None')
    
    # return updated data
    return data

'''
Transforms missing features in training data set only.
'''
def transform_special_train_data(data):
    # Fill missing Electrical with mode value
    data = de.fill_feature_missing_value(data, 'Electrical', data['Electrical'].mode()[0])
    return data

'''
Transforms missing features in testing data set only.
'''
def transform_special_test_data(data):
    
    # BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF
    bsmt_cont_features = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
    for bsmt_feature in bsmt_cont_features:
        data = de.fill_feature_missing_value(data, bsmt_feature, 0.0)

    # BsmtFullBath, BsmtHalfBath
    data = de.fill_feature_missing_value(data, 'BsmtFullBath', 0.0)
    data = de.fill_feature_missing_value(data, 'BsmtHalfBath', 0.0)

    # Exterior1st, Exterior2nd
    data = de.fill_feature_missing_value(data, 'Exterior1st', data['Exterior1st'].mode()[0])
    data = de.fill_feature_missing_value(data, 'Exterior2nd', data['Exterior2nd'].mode()[0])

    # Functional
    data = de.fill_feature_missing_value(data, 'Functional', data['Functional'].mode()[0])

    # GarageCars, GarageArea
    data = de.fill_feature_missing_value(data, 'GarageCars', 0.0)
    data = de.fill_feature_missing_value(data, 'GarageArea', 0.0)

    # KitchenQual
    data = de.fill_feature_missing_value(data, 'KitchenQual', data['KitchenQual'].mode()[0])

    # Fill MSZoning with mode value
    data = de.fill_feature_missing_value(data, 'MSZoning', data['MSZoning'].mode()[0])

    # SaleType
    data = de.fill_feature_missing_value(data, 'SaleType', data['SaleType'].mode()[0])

    # Fill Utilities with mode values
    data = de.fill_feature_missing_value(data, 'Utilities', data['Utilities'].mode()[0])

    # return data
    return data

'''
Transforms and fills NA values in features data.
Features considered are the ones missing at least 1 data point.
'''
def transform_missing_feature_values(data):
    # print sorted(data.columns)
    # fill NA with 0.0 for TotalBsmtSF
    # print "TotalBsmtSF missing:", data['TotalBsmtSF'].isnull().sum()
    if (data['TotalBsmtSF'].any()):
        data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(0.0)

    # fill NA with 0.0 for BsmtFinSF1
    if (data['BsmtFinSF1'].any()):
        data['BsmtFinSF1'] = data['BsmtFinSF1'].fillna(0.0)

    # fill NA with 0.0 for BsmtFinSF2
    if (data['BsmtFinSF2'].any()):
        data['BsmtFinSF2'] = data['BsmtFinSF2'].fillna(0.0)

    # fill NA with 0.0 for BsmtUnfSF
    if (data['BsmtUnfSF'].any()):
        data['BsmtUnfSF'] = data['BsmtUnfSF'].fillna(0.0)

    # fill NA with mode of set for Electrical
    # print "Electrical missing:", data['Electrical'].isnull().sum()
    if (data['Electrical'].any()):
        data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])

    # fill NA with mode of set for Exterior1st
    if (data['Exterior1st'].any()):
        data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])

    # fill NA with mode of set for Exterior2nd
    if (data['Exterior2nd'].any()):
        data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])

    # fill NA with mode of set for KitchenQual
    if (data['KitchenQual'].any()):
        data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])

    # fill NA with 0 for GarageCars
    if (data['GarageCars'].any()):
        data['GarageCars'] = data['GarageCars'].fillna(0)

    # fill NA with 0 for GarageArea
    if (data['GarageArea'].any()):
        data['GarageArea'] = data['GarageArea'].fillna(0.0)

    # fill NA with mode of set for SaleType
    if (data['SaleType'].any()):
        data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])

    # return transformed data
    return data

'''
Entry point.
'''
if __name__ == '__main__':
    main()

