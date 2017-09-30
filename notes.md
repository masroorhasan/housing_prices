* normalize data set
    * feature selection
    * feature normalization
    * plots to figure out
    * cross validate

* rmse score
* compare decision tree regressor to another model
* figure out params for grid search

* heat map "large spots"
    * TotalBsmtSF
    * 1stFlrSF
    
    * GarageYrBlt
    * GarageCars
    * GarageArea

* correlation with SalePrice
    * OverallQual=0.79
    * GrLivArea=0.71
    * GarageCars=0.64
    * GarageArea=0.62
    * TotalBsmtSF=0.61
    * 1stFlrSF=0.61
    * FullBath=0.56
    * TotRmsAbvGrd=0.53
    * YearBuilt=0.52
* Feature selection => remove features with > 1 missing data
* Feature transformation 
    * fill NAs for features with 1 missing data point
    * log transform SalesPrice
* Models
    * DecisionTree
    * RandomForest
    * GradientBoost
* Grid search and cross validate to tune params and get best model
    * TODO: Tune params
* Ensemble models (i.e. take mean of predictions)
* TODO
    * weigh relevant features more? Model from grid search: use best_features to select features and train entire data set on best_estimator
    * improve models with param tuning
    * re-fit entire train set using best_estimator params (after cross validation)