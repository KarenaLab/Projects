# [P504] Cement compressive strength

# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (mean_absolute_error, root_mean_squared_error,
                             r2_score)

import matplotlib.pyplot as plt


# Personal modules
from concrete_tools import (load_dataset, cols_variable, prep_pipeline,
                            aggregate_results)


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


def model_linregr(x_train, x_test, y_train, y_test=None):
    regr = LinearRegression()

    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    params = dict()
    params["coef"] = regr.coef_
    params["intercept"] = regr.intercept_

    return y_pred, params


def regr_metrics(y_true, y_pred):
    results = dict()
    results["mae"] = mean_absolute_error(y_true, y_pred)
    results["rmse"] = root_mean_squared_error(y_true, y_pred)
    results["r2_score"] = r2_score(y_true, y_pred)
    results["pearson_r"] = stats.pearsonr(y_true, y_pred).statistic

    return results


def print_results(dictionary, decimals=3):
    seed = results["seed"]
    mae = np.round(results["mae"], decimals=decimals)
    rmse = np.round(results["rmse"], decimals=decimals)
    r2 = np.round(results["r2_score"], decimals=4)
    pearson = np.round(results["pearson_r"], decimals=4)

    print("  Seed      MAE     RMSE   R2 Score    Pearson")
    print(f"{str(seed):>6s}{str(mae):>9s}{str(rmse):>9s}{str(r2):>11s}{str(pearson):>11s}")

    return None

                     
# Setup/Config


    
# Program --------------------------------------------------------------
df = load_dataset()
target = "compressive_strength_mpa"
seed = 15

# Data Split
x, y = split_target(df, target=target)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,
                                                    random_state=seed)

# Model: Linear Regression
x_train, x_test = scaler(x_train, x_test)
y_pred, _ = model_linregr(x_train, x_test, y_train)

# Results
results = regr_metrics(y_test, y_pred)
results["seed"] = seed

print_results(results)

# Seed      MAE     RMSE   R2 Score    Pearson
#   42    8.985   11.235     0.5608     0.7494   
#   53    8.082   10.195     0.5896     0.7654
#   27    8.182   10.494     0.5794     0.7613
#   15    7.792    9.901     0.6167     0.7853
