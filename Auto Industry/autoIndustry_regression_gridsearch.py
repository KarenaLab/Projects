
# Libraries
import os

import numpy as np
import pandas as pd


# Personal modules
import sys
sys.path.append(r"C:\python_modules")

from autoindustry_tools import *

from dataframe_preparation import *
from split_kfold import *
from normalization_builtin import *
from regr_linearregression import *
from results_analysis import *
from print_results import *

from plot_histogram import *
from plot_scatterhistlinreg import *
from plot_boxplot import *
from plot_identityline import *
from plot_blandaltman import *


# Functions
def results_preparation(array):

    data = results_to_dataframe(flatten_list(array))
    data = data[["model", "fit_intercept", "positive", "fold",
                 "pearson", "mae", "rmse", "bias", "r2_score"]]

    data = data.sort_values(by=["model", "fit_intercept", "positive", "fold"])
    

    return data


# Setup/Config
seed = 302
target = "power_hp"

n_splits = 5
metrics = ["mae", "rmse", "smape", "bias", "r2_score", "pearson"]
savefig = False


# Program --------------------------------------------------------------
filename = "auto_industry.csv"
df = read_csv(filename)

# Data Preparation: Very simple (without origin and name)
cols_remove = ["origin", "name", "model_year"]
df = df.drop(columns=cols_remove)
df = dataframe_preparation(df)
df = units_conversion(df)

# Featuring Engineering
df["weight_power_ratio"] = np.round(df["weight_kg"] / df["power_hp"], decimals=3)

# Target split
x, y = split_target(df, target=target)

np.random.seed(seed)
split_table = split_kfold(x, n_splits=n_splits)


# Model Base = Linear Regression
results = list()
for i in range(0, n_splits):
    fold, train_index, test_index = split_table[i]
    x_train, y_train, x_test, y_test = separate_fold(x, y, train_index, test_index)

    # Scaler = Standard Score
    x_train, x_test = scaler_standardscore(x_train, x_test)

    # Model = Grid Search with Linear Regression
    fold_results = gridsearch_linreg(x_train, y_train, x_test, y_test,
                                     add_to_results={"fold": fold}, metrics=metrics)

    results.append(fold_results)


for i in range(0, n_splits):
    fold, train_index, test_index = split_table[i]
    x_train, y_train, x_test, y_test = separate_fold(x, y, train_index, test_index)

    # Scaler = Standard Score
    x_train, x_test = scaler_standardscore(x_train, x_test)

    # Model = Grid Search with Linear Regression
    fold_results = gridsearch_ridge(x_train, y_train, x_test, y_test,
                                    alpha=[0.01, 0.5, 1, 2, 5, 10],
                                    fit_intercept=[True], positive=[False],
                                    add_to_results={"fold": fold}, metrics=metrics)

    results.append(fold_results)   


# Results from a list of dictionaries to pandas DataFrame
#results = results_preparation(results)


"""
# Results Analysis
fit_intercept = results["fit_intercept"].unique()
positive = results["positive"].unique()

best_fi, best_p = "", ""
best_rmse = np.inf
best_mae = np.inf

for fi, p in itertools.product(fit_intercept, positive):
    data = results.loc[(results["fit_intercept"] == fi) &
                       (results["positive"] == p)]

    rmse_mean = data["rmse"].mean()
    mae_mean = data["mae"].mean()

    if(rmse_mean < best_rmse):
        best_fi = fi
        best_p = p
        best_rmse = rmse_mean
        best_mae = mae_mean

        print("change")
        print(fi, p, rmse_mean, mae_mean)


print(f" > Best parameters: fit_intercept:{best_fi}, positive:{best_p}")
"""
