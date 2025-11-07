# [P504] Cement compressive strength
# Linear Regression model with seed variation

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
                            aggregate_results, organize_report)

from src.plot_histbox import plot_histbox


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


def pipeline(DataFrame, target, test_size=0.20, random_state=None):
    # Data split
    x, y = split_target(DataFrame, target=target)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    x_train, x_test = scaler(x_train, x_test)

    # Model: Linear Regression
    y_pred, _ = model_linregr(x_train, x_test, y_train)
    results = regr_metrics(y_test, y_pred)

    return results
    
    
                  
# Setup/Config
savefig = False

    
# Program --------------------------------------------------------------
df = load_dataset()
target = "compressive_strength_mpa"

# Data Split
seed = 1
np.random.seed(seed)

df_results = pd.DataFrame(data=[])

size = 100
for seed in np.random.randint(low=0, high=500, size=size):
    results = pipeline(df, target=target, test_size=0.30, random_state=seed)

    for key, _ in results.items():
        df_results.loc[seed, key] = results[key]


for col in df_results.columns:
    plot_histbox(df_results[col], title=f"Concrete Strength - Linear Regression v01 - {col}",
                 savefig=savefig)


# Scout theme
# "Always leave the campsite cleaner than you found it"
organize_report()
