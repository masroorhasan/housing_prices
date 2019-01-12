# import python modules
import pandas as pd
import numpy as np
import datetime
import joblib

# import modules
import data_exploration as de

# import sklearn modules
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_best_gb_regressor_model(X, y):
    '''
    Gets best GradientBoost regressor model based on grid search, trained on data [X, y].
    '''
    # Create cross validation sets from training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.20, random_state=42)

    # Create the gradient boost regressor object
    regressor = GradientBoostingRegressor(random_state=42)

    # Params to tune
    gs_params = {
        'n_estimators':[2000,3000],
        'learning_rate':[0.01,0.05],
        'max_depth':[3,4,5],
        'min_samples_leaf':[20,26],
        'min_samples_split':[2,5,10]
        # 'max_leaf_nodes':[None,5]
    }

    # Use r2 as scoring function
    gs_scoring_func = make_scorer(perf_metric_r2)

    # Create grid search object and fit the data
    grid_search = GridSearchCV(regressor, param_grid=gs_params, scoring=gs_scoring_func, cv=cv_sets)
    model = grid_search.fit(X, y)

    # Print optimal params
    print "GradientBoosting"
    print model.best_params_

    # return the model
    return model

def get_best_rf_regressor_model(X, y):
    '''
    Gets best RandomForest regressor model based on grid search, trained on data [X, y].
    '''
    # Create cross validation sets from training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.20, random_state=42)

    # Create the gradient boost regressor object
    regressor = RandomForestRegressor(n_jobs=5, random_state=42)

    # Params to tune
    gs_params = {
        'n_estimators':[180,190,200],
        'max_depth':[10,12,15],
        'min_samples_leaf':[2,3],
        'min_samples_split':[5,10],
    }

    # Use r2 as scoring function
    gs_scoring_func = make_scorer(perf_metric_r2)

    # Create grid search object and fit the data
    grid_search = GridSearchCV(regressor, param_grid=gs_params, scoring=gs_scoring_func, cv=cv_sets)
    model = grid_search.fit(X, y)

    # Print optimal params
    print "RandomForest"
    print model.best_params_

    # Return the model
    return model

def print_model_prediction_scores(model, X, y):
    '''
    Prints model prediction scores based on [X, y] data.
    '''
    # Print model details
    print(model)

    # Precit from set
    predict = model.predict(X)
    print "prediction score of cv set:"
    print "R2 score: {0:.3f}".format(perf_metric_r2(y, predict))
    print "RMSE score: {0:.3f}".format(perf_metric_rmse(y, predict))
    # de.pred_scatter_plot(y, predict)
    print "\n"

def perf_metric_r2(y_true, y_predict):
    '''
    Calculate and return r2 score.
    '''
    # Calculate r2 score and print
    r2 = r2_score(y_true, y_predict)
    # print "r2 score: {0:.3f}".format(r2)
    return r2

def perf_metric_rmse(y_true, y_predict):
    '''
    Calculate and return RMSE.
    '''
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_predict))
    # print "RMSE score: {0:.3f}".format(rmse)
    return rmse

def transform_features(training_set, testing_set):
    '''
    Transforms features of training and testing sets.
    Returns concatanated data set.
    '''
    # Features missing values in training set
    print "Missing data in TRAIN features"
    missing_train = sorted(de.missing_features(training_set))
    print missing_train

    # Features missing values in testing set
    print "Missing data in TEST features"
    missing_test = sorted(de.missing_features(testing_set))
    print missing_test

    # Features missing in testing and training sets
    print "Intersect:"
    print sorted(list(set(missing_test) & set(missing_train)))

    # Features missing only in testing set
    print "Difference:"
    print sorted(list(set(missing_test) - set(missing_train)))

    # Transform features in training set by filling in NaN values
    print "Before filling in missing TRAIN data features:"
    print training_set[de.missing_features(training_set)].isnull().sum()
    training_set = transform_missing_feature_data(training_set)
    training_set = transform_special_train_data(training_set)
    print "After filling in missing TRAIN data features"
    print training_set[de.missing_features(training_set)].isnull().sum()

    # Transform features in testing set by filling in NaN values
    print "Before filling in missing TEST data features:"
    print testing_set[de.missing_features(testing_set)].isnull().sum()
    testing_set = transform_missing_feature_data(testing_set)
    testing_set = transform_special_test_data(testing_set)
    print "After filling in missing TEST data features"
    print testing_set[de.missing_features(testing_set)].isnull().sum()

    # Combine training and testing sets into df
    concat_data_set = pd.concat([training_set, testing_set], keys=['train', 'test'])

    # Fix features that should be categorical
    concat_data_set = transform_num_to_cat(concat_data_set)

    # Transform categorical features by applying label encoding
    concat_data_set = transform_label_encoding(concat_data_set)

    # Return transformed set
    return concat_data_set

def transform_missing_feature_data(data):
    '''
    Transforms missing feature values in both training and testing data set.
    '''
    # Fill NA with 'None' for categorical missing features
    data = de.fill_feature_missing_value(data, 'Alley', 'None')

    # Bsmt* missing features
    bsmt_features = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']
    # Missing basement features have same indices where total basement square footage is 0.0 (i.e. no basement)
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

def transform_special_train_data(data):
    '''
    Transforms missing features in training data set only.
    '''
    # Fill missing Electrical with mode value
    data = de.fill_feature_missing_value(data, 'Electrical', data['Electrical'].mode()[0])
    return data

def transform_special_test_data(data):
    '''
    Transforms missing features in testing data set only.
    '''
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

def transform_num_to_cat(data):
    '''
    Transforms numerical data that should be categorical.
    '''
    # MSSubClass
    data['MSSubClass'] = data['MSSubClass'].apply(str)

    # MoSold
    data['MoSold'] = data['MoSold'].astype(str)

    # OverallCond
    data['OverallCond'] = data['OverallCond'].astype(str)

    # YrSold
    data['YrSold'] = data['YrSold'].astype(str)

    # Return the data
    return data

def transform_label_encoding(data):
    '''
    Tansforms categorical data values with label encoding.
    '''
    # Get features with categorical data
    cat_features = sorted(list(set(data.columns) - set(data._get_numeric_data().columns)))

    # Process each categorical feature and apply LabelEncoder to it
    for feature in cat_features:
        label_encoder = LabelEncoder()
        label_encoder.fit(list(data[feature].values))
        data[feature] = label_encoder.transform(list(data[feature].values))
    return data

def train_gradient_boost(featues, labels, X_train, X_test, y_train, y_test):
    """
    Trains a GradientBoosting regressor model
    """
    gb_params = {
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'max_features':'sqrt',
        'loss':'huber',
        'max_depth': 3,
        'min_samples_split': 10,
        'min_samples_leaf': 25
        }
    gb_regressor_model = GradientBoostingRegressor(random_state=42, **gb_params)
    gb_regressor_model = gb_regressor_model.fit(X_train, y_train)
    print_model_prediction_scores(gb_regressor_model, X_test, y_test)
    # create new gradientboosting regressor with params and train against full set
    best_gb_regressor = GradientBoostingRegressor(random_state=42, **gb_params)
    best_gb_regressor.fit(featues, labels)
    return best_gb_regressor

def train_random_forest(featues, labels, X_train, X_test, y_train, y_test):
    """
    Trains a RandomForest regressor model
    """
    rf_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        }
    rf_regressor_model = RandomForestRegressor(random_state=42, n_jobs=5, **rf_params)
    rf_regressor_model = rf_regressor_model.fit(X_train, y_train)
    print_model_prediction_scores(rf_regressor_model, X_test, y_test)
    # create new randomforest regressor with params and train against full set
    best_rf_regressor = RandomForestRegressor(random_state=42, n_jobs=5, **rf_params)
    best_rf_regressor.fit(featues, labels)
    return best_rf_regressor

def ensemble_predict(features, testing_set, best_gb_regressor, best_rf_regressor):
    """
    Ensemble and predict
    Predict on testing set and save to csv
    """
    testing_features = features.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
    pred_test_set = (
        np.exp(best_gb_regressor.predict(testing_features)) +
        np.exp(best_rf_regressor.predict(testing_features))
        ) / 2
    filename = "%s.csv" % (datetime.datetime.now().strftime("%Y-%m-%d,%H:%M:%S"))
    pd.DataFrame({'Id': testing_set.Id, 'SalePrice':pred_test_set}).to_csv(filename, index=False)

def save_model(model, model_file):
    joblib.dump(model, model_file)
    print("Model export success {}".format(model_file))

def main():
    # Build training and testing tests
    training_set = pd.read_csv('data/train.csv')
    testing_set = pd.read_csv('data/test.csv')
    print training_set['SalePrice'].describe()

    # Get SalePrice labels from training set
    training_labels = training_set.pop('SalePrice')
    training_labels = np.log(training_labels)   # normalize with log transform

    # Feature transformation
    features = transform_features(training_set, testing_set)
    training_features = features.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

    # split data for train and test cross validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        training_features,
        training_labels,
        test_size=0.2,
        random_state=42)


    ## GradientBoosting regressor ##
    gb_regressor = train_gradient_boost(training_features,
                                        training_labels,
                                        X_train,
                                        X_test,
                                        y_train,
                                        y_test)

    ##
    ## RandomForest regressor ##
    rf_regressor = train_random_forest(training_features,
                                        training_labels,
                                        X_train,
                                        X_test,
                                        y_train,
                                        y_test)

    ## save models
    save_model(gb_regressor, 'gb_regressor.dat')
    save_model(rf_regressor, 'rf_regressor.dat')

if __name__ == '__main__':
    main()
