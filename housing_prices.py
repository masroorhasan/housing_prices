# import data modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    print training_set.columns
    print training_set['SalePrice'].describe()

    # scatter plot grlivarea/saleprice
    # data = pd.concat([training_set['SalePrice'], training_set['OverallQual']], axis=1)
    # data.plot.scatter(x='OverallQual', y='SalePrice', ylim=(0,800000));
    

    # correlation matrix
    corrmat = training_set.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True);
    plt.show()

    # split training set to features and labels
    training_set_labels = training_set['SalePrice']
    training_set_features = training_set.drop('SalePrice', axis=1)
    
    # TODO: clean up and preprocess data
    # 1. drop irrelevant/invalid features 
        # remove features with null values
    # 2. label encoding on features requiring transformation

    # test with a few relevant features only
    relevant_features = ['OverallQual', 'OverallCond', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea']
    training_set_features = training_set_features[relevant_features]

    # shuffle and split training data
    X_train, X_test, y_train, y_test = train_test_split(
        training_set_features, 
        training_set_labels, 
        test_size=0.2, 
        random_state=42)
    
    # get optimal decision tree regressor model
    regressor_model = get_regressor_model(X_train, y_train)
    print "Optimal max depth for model {}".format(regressor_model.get_params()['max_depth'])

    # predict on test sets
    pred = regressor_model.predict(X_test)
    # perf_metric_rmse(y_test, pred)
    print "R2 score from model prediction: {0:.3f}".format(perf_metric_r2(y_test, pred))

'''
Performs grid search cross validation over max depth parameter to get optimal decision tree regressor model
trained on input data [X, y].
'''
def get_regressor_model(X, y):
    # create cross validation sets from training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.20, random_state=42)

    # create decision tree regressor object
    regressor = DecisionTreeRegressor()

    # create dictionary for max_depth param with range [0,10]
    gs_params = {'max_depth':range(1,11)}

    # use r2 as scoring function
    gs_scoring_func = make_scorer(perf_metric_r2)

    # create grid search object
    grid_search = GridSearchCV(regressor, param_grid=gs_params, scoring=gs_scoring_func, cv=cv_sets)

    # compute optimal model by fitting grid search object to data
    grid_search = grid_search.fit(X, y)

    # return optimal model from fitted data
    return grid_search.best_estimator_

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
    rmse = mean_squared_error(y_true, y_predict)
    # print "RMSE score: {0:.3f}".format(rmse)
    return rmse
    
'''
Entry point.
'''
if __name__ == '__main__':
    main()

