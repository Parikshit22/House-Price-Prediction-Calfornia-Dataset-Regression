# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 12:16:01 2018

@author: MUJ
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator , TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv('housing.csv')


print(dataset.shape)
# head
print(dataset.head(20))
# descriptions
print(dataset.describe())

dataset["ocean_proximity"].value_counts()



# method first to split taraning data with test data using numpy
#def split_train_test(data,test_ratio):
#    np.random.seed(7)
#    shuffled_indices = np.random.permutation(len(data))
#    test_set_size = int(len(data)* test_ratio)
#    test_indices = shuffled_indices[:test_set_size]
#    train_indices = shuffled_indices[test_set_size:]
#    return data.iloc[train_indices], data.iloc[test_indices]


#train_set,test_set = split_train_test(dataset,0.2)
#print(len(train_set),"train+",len(test_set),"test")


# method second to split taraning data with test data using sklearn
#train_set,test_set = train_test_split(dataset, test_size = 0.2 , random_state = 42)
#print(len(train_set),"train+",len(test_set),"test")
#print(train_set)

## method third to split taraning data with test data
dataset["income_cat"] = np.ceil(dataset["median_income"]/1.5)
dataset["income_cat"].where(dataset["income_cat"] < 5,5.0,inplace= True)

split = StratifiedShuffleSplit(n_splits=1,test_size =0.2, random_state=42)
for train_index, test_index in split.split(dataset,dataset["income_cat"]):
    start_train_set = dataset.loc[train_index]
    start_test_set =  dataset.loc[test_index]

dataset = start_train_set.copy()


dataset.plot(kind = "scatter",x="longitude",y="latitude")
dataset.plot(kind = "scatter",x="longitude",y="latitude",alpha = 0.4,
            s=dataset["population"]/100,label="population",figsize=(10,7),
            c="median_house_value",cmap=plt.get_cmap("gist_rainbow"),colorbar=True)
plt.legend()


corr_matrix = dataset.corr()
print(corr_matrix["median_house_value"].sort_values(ascending = False))

attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(dataset[attributes],figsize=(8,8))

dataset["rooms_per_household"] = dataset["total_rooms"]/dataset["households"]
dataset["bedrooms_per_room"] = dataset["total_bedrooms"]/dataset["total_rooms"]
dataset["population_per_household"] = dataset["population"]/dataset["households"]
corr_matrix = dataset.corr()
print(corr_matrix["median_house_value"].sort_values(ascending = False))

dataset1 = dataset
dataset_labels = dataset["median_house_value"].copy()
dataset = dataset.drop("median_house_value",axis=1)
dataset = dataset.dropna(subset = ["total_bedrooms"])

median = dataset["total_bedrooms"].median()
dataset["total_bedrooms"].fillna(median, inplace=True)
"""
imputer = Imputer(strategy = "median")
dataset_num = dataset.drop("ocean_proximity",axis=1)
imputer.fit(dataset_num)
print(dataset.describe())

x= imputer.transform(dataset_num)
dataset_tr = pd.DataFrame(x, columns = dataset_num.columns)
print(dataset.describe())
"""
encoder = LabelEncoder()
dataset_cat = dataset["ocean_proximity"]
dataset_cat_encoded = encoder.fit_transform(dataset_cat)
print(dataset_cat_encoded)

encoder = OneHotEncoder()
dataset_cat_hot = encoder.fit_transform(dataset_cat_encoded.reshape(-1,1))
print(dataset_cat_hot.toarray())

print(dataset.describe())

num_pipeline = Pipeline([('imputer',Imputer(strategy="median")),
                         ('std_scaler',StandardScaler())])
dataset_num_tr = num_pipeline.fit_transform(dataset_num)
print(dataset_num_tr)


class DataFrameSelector(BaseEstimator , TransformerMixin):
    def __init__(self,attribut_names):
        self.attribut_names = attribut_names
    def fit(self , X , y=None):
        return self
    def transform(self,X):
        return X[self.attribut_names].values

num_attribs = list(dataset_num)   
cat_attribs =["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector' , DataFrameSelector(num_attribs)),(
        'imputer', Imputer(strategy = "median")),
          ('std_scaler',StandardScaler())]) 
cat_pipeline = Pipeline([
        ('selector' , DataFrameSelector(num_attribs)),
        ])

full_pipeline = FeatureUnion(transformer_list = [
        ("num_pipeline",num_pipeline),
        ("cat_pipeline",cat_pipeline)])
dataset_prepared = full_pipeline.fit_transform(dataset)
print(dataset_prepared)
print(dataset_prepared.shape)

lin_reg = LinearRegression()
lin_reg.fit(dataset_prepared,dataset_labels)


some_data = dataset.iloc[:5]
some_labels = dataset_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("prediction: ", lin_reg.predict(some_data_prepared))
print("Labels: ", list(some_labels))

dataset_prediction = lin_reg.predict(dataset_prepared)
lin_mse = mean_squared_error(dataset_labels,dataset_prediction)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(dataset_prepared,dataset_labels)
dataset_prediction = tree_reg.predict(dataset_prepared)
tree_mse = mean_squared_error(dataset_labels,dataset_prediction)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

scores = cross_val_score(tree_reg,dataset_prepared,dataset_labels,scoring = 
                         "neg_mean_squared_error",cv = 10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard deviation:" , scores.std())
    
display_scores(tree_rmse_scores)    

forest_reg = RandomForestRegressor()
forest_reg.fit(dataset_prepared,dataset_labels)

dataset_prediction = forest_reg.predict(dataset_prepared)
forest_mse = mean_squared_error(dataset_labels,dataset_prediction)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)
scores = cross_val_score(forest_reg,dataset_prepared,dataset_labels,scoring = 
                         "neg_mean_squared_error",cv = 10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)

param_grid = [
        {'n_estimators': [3,10,30], 'max_features':[2,4,6,8]},
        {'bootstrap': [False], 'n_estimators':[3,10], 'max_features':[2,3,4]},]
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg,param_grid,cv=5,scoring = 
                         'neg_mean_squared_error')
grid_search.fit(dataset_prepared, dataset_labels)
grid_search.best_estimator_

cvres = grid_search.cv_results_
for mean_score,params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

   