* normalize data set
    * feature selection
    * feature normalization
    * plots to figure out
    * cross validate

* rmse score
* compare decision tree regressor to another model
* figure out params for grid search

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

* Feature selection => remove features with > 1 missing data? (TODO: play around with this)

* Feature transformation 
    * fill NAs for features with 1 missing data point
    * log transform SalesPrice
    
* Grid search and cross validate to tune params and get best model
    * tune params for GradientBoost regressor
    * tune params for RandomForest
    * should tree based params be the same? (TODO)

* Model building
    * DecisionTree - terrible
    * RandomForest
    * GradientBoost
    * AdaBoost?
    * use best_features from optimal model (grid search and cross validation) to re-fit entire data set on new model instance
    * improve models with param tuning

* Final predictions
    * Take average prediction value of (optimal) models trained on full training data set