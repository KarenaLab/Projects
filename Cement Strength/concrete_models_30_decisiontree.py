# [P504] Cement compressive strength
# Performs EDA (Exploratory Data Analysis)

# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (mean_absolute_error, root_mean_squared_error,
                             r2_score)

import matplotlib.pyplot as plt


# Personal modules
from concrete_tools import (load_dataset, cols_variable, prep_pipeline,
                            aggregate_results, organize_report)

from src.plot_linemultiple import plot_linemultiple


# Functions
def split_target(DataFrame, target):
    x = DataFrame.drop(columns=[target])
    y = DataFrame[target]

    return x, y


def scaler(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)

    cols = x_train.columns
    for i in [x_train, x_test]:
        i = scaler.transform(i)
        i = pd.DataFrame(data=i, columns=cols)

    return x_train, x_test


def regr_decisiontree(x_train, x_test, y_train,
                      max_depth=None, max_features=None, max_leaf_nodes=None,
                      random_state=None):

    # Model: Decison Tree
    regr = DecisionTreeRegressor()

    # Hyperparams
    regr.max_depth = max_depth
    regr.max_features = max_features
    regr.max_leaf_nodes = max_leaf_nodes
    regr.random_state = random_state

    # Fit and predict
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    # Parameters
    params = dict()

    return y_pred, params
    

def regr_kneigbors(x_train, x_test, y_train, n_neighbors=2):
    regr = KNeighborsRegressor()
    # Main parameters: n_neighbors, weights, p,

    # Hyperparams
    regr.n_neighbors = n_neighbors

    # Fit and predict
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    # Parameters
    params = dict()


    return y_pred, params
   

def regr_metrics(y_true, y_pred):
    results = dict()
    results["mae"] = mean_absolute_error(y_true, y_pred)
    results["rmse"] = root_mean_squared_error(y_true, y_pred)
    results["r2_score"] = r2_score(y_true, y_pred)
    results["pearson_r"] = stats.pearsonr(y_true, y_pred).statistic

    return results


def find_best_hyperp(DataFrame, metric, best):
    """


    """
    # Data preparation
    DataFrame = DataFrame.dropna()

    # Set best parameters   
    if(best == "lower"):
        best_metric = np.inf

    elif(best == "upper"):
        best_metric = 0

    # Finder
    best_param = None

    for i in DataFrame.index:
        value = DataFrame.loc[i, metric]
        if(best == "lower" and value < best_metric):
            best_metric = value
            best_param = i

        elif(best == "upper" and value > best_metric):
            best_metric = value
            best_param = i

        else:
            break


    return best_param 

                     
# Setup/Config
savefig = False

    
# Program --------------------------------------------------------------
df = load_dataset()
target = "compressive_strength_mpa"

# Data Split
x, y = split_target(df, target=target)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,
                                                    random_state=27)

# Scaler
x_train, x_test = scaler(x_train, x_test)

# Model: Decision Tree
y_pred, _ = regr_decisiontree(x_train, x_test, y_train,
                              max_depth=None, max_features=None, max_leaf_nodes=None,
                              random_state=314)
    
results = regr_metrics(y_test, y_pred)    

                 
# Scout theme: "Always leave the campsite cleaner than you found it"
organize_report()

