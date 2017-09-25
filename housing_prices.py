# import data modules
import pandas as pd
import numpy as np

# import sklearn modules
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.cross_validation import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

'''
Main method.
'''
def main():
    # build training and testing tests
    training_set = pd.read_csv('data/train.csv')
    testing_set = pd.read_csv('data/test.csv')
    # print training_set.head()
    # print testing_set.head()
    
    # TODO: clean up data:
    # 1. drop irrelevant/invalid features
    # 2. label encoding on features requiring transformation

    # split training set to features and labels
    training_set_labels = training_set['SalePrice']
    training_set_features = training_set.drop('SalePrice', axis=1)

    # find best decision tree regressor model with grid search and cross validation
    regressor_model = fit_model(training_set_features, training_set_labels)
    print "Optimal max depth for model {}".format(regressor_model.get_params()['max_depth'])

'''
Performs grid search cross validation over max depth parameter for a decision tree regressor model
trained on input data [X, y].
'''
def fit_model(X, y):
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
    print "r2 score: {0:.3f}".format(r2)
    return r2

'''
Calculate and return RMSE.
'''
def perf_metric_rmse(y_true, y_predict):
    # calculate RMSE 
    rmse = mean_squared_error(y_true, y_predict)
    print "RMSE score: {0:.3f}".format(rmse)
    return rmse
    
'''
Entry point.
'''
if __name__ == '__main__':
    main()

