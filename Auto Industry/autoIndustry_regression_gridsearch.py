
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
def results_analytics(DataFrame, beta=1, verbose=True):
    """


    Return: best model, best fit_intercept, best positive and best alpha.

    """
    # Variables selection
    model = DataFrame["model"].unique()
    fit_intercept = DataFrame["fit_intercept"].unique()
    positive = DataFrame["positive"].unique()
    alpha = DataFrame["alpha"].unique()

    # Reset variables
    best_m = np.nan
    best_fi = np.nan
    best_p = np.nan
    best_a = np.nan
    best_score = np.inf

    # Find the best hiperparameters (Grid Search) using F-be
    for m, fi, p, a in itertools.product(model, fit_intercept, positive, alpha):
        data = results.loc[(DataFrame["model"] == m) &
                           (DataFrame["fit_intercept"] == fi) &
                           (DataFrame["positive"] == p) &
                           (DataFrame["alpha"] == a)]

        # Calculate RMSE and MAE from Grid Search
        rmse = data["rmse"].mean()
        mae = data["mae"].mean()
        score = fb_score(rmse, mae, beta=beta)

        if(score < best_score):
            best_m = m
            best_fi = fi
            best_p = p
            best_a = a
            best_score = score

    # Print values
    if(verbose == True):
        print(f" > Best parameters: model:{best_m}, fit_intercept:{best_fi}, positive:{best_p}, alpha:{best_a}")


    return best_m, best_fi, best_p, best_a 



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

# Results from a list of dictionaries to pandas DataFrame
results = results_to_dataframe(results)

# Select best parameters for regularized models.
_, best_fi, best_p, _ = results_analytics(results)


# Ridge (L2)
results = list()

for i in range(0, n_splits):
    fold, train_index, test_index = split_table[i]
    x_train, y_train, x_test, y_test = separate_fold(x, y, train_index, test_index)

    # Scaler = Standard Score
    x_train, x_test = scaler_standardscore(x_train, x_test)

    # Model = Grid Search with Ridge Regression
    fold_results = gridsearch_ridge(x_train, y_train, x_test, y_test,
                                    alpha=[0.01, 0.5, 1, 2, 5, 10],
                                    fit_intercept=[best_fi], positive=[best_p],
                                    add_to_results={"fold": fold}, metrics=metrics)

    results.append(fold_results)


# Lasso (L1)
for i in range(0, n_splits):
    fold, train_index, test_index = split_table[i]
    x_train, y_train, x_test, y_test = separate_fold(x, y, train_index, test_index)

    # Scaler = Standard Score
    x_train, x_test = scaler_standardscore(x_train, x_test)

    # Model = Grid Search with Lasso Regression
    fold_results = gridsearch_lasso(x_train, y_train, x_test, y_test,
                                    alpha=[0.01, 0.5, 1, 2, 5, 10],
                                    fit_intercept=[best_fi], positive=[best_p],
                                    add_to_results={"fold": fold}, metrics=metrics)

    results.append(fold_results)


# Results from a list of dictionaries to pandas DataFrame
results = results_to_dataframe(results)

# Select best parameters for regularized models.
best_model, best_alpha, best_fi, best_p = results_analytics(results)
