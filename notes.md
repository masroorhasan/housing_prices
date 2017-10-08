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

-- ~87% R2 score


**Missing Data** 

['Alley', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF', 'Electrical', 'Exterior1st', 'Exterior2nd', 'Fence', 'FireplaceQu', 'Functional', 'GarageArea', 'GarageCars', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt', 'KitchenQual', 'LotFrontage', 'MSZoning', 'MasVnrArea', 'MasVnrType', 'MiscFeature', 'PoolQC', 'SaleType', 'TotalBsmtSF', 'Utilities']

-Alley: Type of alley access to property (convert NA to None enum?)
-Bsmt*: Related to BsmtFinSF1 and BsmtFinSF2? (NA means no basement)
-Electrical: Electrical system (take mode)
-Fence: Fence quality (convert NA to None enum?)
-FireplaceQu: Fireplace quality (convert NA to None enum)
-Garage*: Related to GarageCars, GarageArea?
-LotFrontage: Linear feet of street connected to property (take mean/mode?)
-MasVnrArea: Masonry veneer area in square feet (take mean/mode?)
-MasVnrType: See MasVnrArea
-MiscFeature: Miscellaneous feature not covered in other categories (convert NA to None enum)
-PoolQC: Pool quality (convert NA to None enum)


Missing data in TRAIN features
['Alley', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'Electrical', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt', 'LotFrontage', 'MasVnrArea', 'MasVnrType', 'MiscFeature', 'PoolQC']

Missing data in TEST features
['Alley', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF', 'Exterior1st', 'Exterior2nd', 'Fence', 'FireplaceQu', 'Functional', 'GarageArea', 'GarageCars', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt', 'KitchenQual', 'LotFrontage', 'MSZoning', 'MasVnrArea', 'MasVnrType', 'MiscFeature', 'PoolQC', 'SaleType', 'TotalBsmtSF', 'Utilities']

Intersect:
['Alley', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt', 'LotFrontage', 'MasVnrArea', 'MasVnrType', 'MiscFeature', 'PoolQC']

Difference:
['BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'Exterior1st', 'Exterior2nd', 'Functional', 'GarageArea', 'GarageCars', 'KitchenQual', 'MSZoning', 'SaleType', 'TotalBsmtSF', 'Utilities']

GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.05, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_split=1e-07,
             min_samples_leaf=10, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=500,
             presort='auto', random_state=42, subsample=1.0, verbose=0,
             warm_start=False)
prediction score of cv set:
R2 score: 0.896
RMSE score: 0.139


RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=15,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=2,
           min_samples_split=5, min_weight_fraction_leaf=0.0,
           n_estimators=190, n_jobs=5, oob_score=False, random_state=42,
           verbose=0, warm_start=False)
prediction score of cv set:
R2 score: 0.882
RMSE score: 0.148
