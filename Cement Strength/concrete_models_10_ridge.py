# [P504] Cement compressive strength
# Performs EDA (Exploratory Data Analysis)

# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

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
    train = scaler.transform(x_train)
    train = pd.DataFrame(data=train, columns=cols)

    cols = x_test.columns
    test = scaler.transform(x_test)
    test = pd.DataFrame(data=test, columns=cols)    

    return train, test


def regr_linregr(x_train, x_test, y_train):
    # Model
    regr = LinearRegression()

    # Fit and predict
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    # Parameters
    params = dict()
    params["coef"] = regr.coef_
    params["intercept"] = regr.intercept_

    return y_pred, params


def regr_ridge(x_train, x_test, y_train, alpha=1):
    regr = Ridge(alpha=1)
    # Main parameters: Alpha*, fit_intercept, positive and
    #                  random_state

    # Hyperparams
    regr.alpha = alpha

    # Fit and predict
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    # Parameters
    params = dict()


    return y_pred, params   


def alpha_range():
    values_10 = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    values_log = [3.16227766e-03, 3.16227766e-02, 3.16227766e-01,
                  3.16227766e+00, 3.16227766e+01, 3.16227766e+02,
                  3.16227766e+03]

    values = values_10 + values_log
    values.sort(reverse=True)

    return values


def regr_metrics(y_true, y_pred):
    results = dict()
    results["mae"] = mean_absolute_error(y_true, y_pred)
    results["rmse"] = root_mean_squared_error(y_true, y_pred)
    results["r2_score"] = r2_score(y_true, y_pred)
    results["pearson_r"] = stats.pearsonr(y_true, y_pred).statistic

    return results


                     
# Setup/Config


    
# Program --------------------------------------------------------------
df = load_dataset()
target = "compressive_strength_mpa"

# Data Split
x, y = split_target(df, target=target)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,
                                                    random_state=27)

# Scaler
x_train, x_test = scaler(x_train, x_test)

df_results = pd.DataFrame(data=[])
# Model: Ridge (L2)
for a in alpha_range():
    y_pred, params = regr_ridge(x_train, x_test, y_train, alpha=a)
    results = regr_metrics(y_test, y_pred)    

    for key, value in results.items():
        df_results.loc[a, key] = value











