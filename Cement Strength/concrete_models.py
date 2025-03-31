# [P504] Cement compressive strength
# Performs EDA (Exploratory Data Analysis)

# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (r2_score, mean_absolute_error)


import matplotlib.pyplot as plt


# Personal modules
from concrete_tools import load_dataset

from src.split_train_test import split_train_test
from src.stratified_continuous_kfold import stratified_continuous_kfold

from src.plot_histbox import plot_histbox


# Functions
def cols_variable():
    cols = ["cement_kg_p_m3", "blast_furnace_slag_kg_p_m3",
            "fly_ash_kg_p_m3", "water_kg_p_m3", "superplasticizer_kg_p_m3",
            "coarse_aggregate_kg_p_m3", "fine_aggregate_kg_p_m3", "age_days"]

    return cols


# Setup/Config


def prep_pipeline(DataFrame, train_index, test_index, target):
    """


    """
    # Split Train and Test with target
    cols_vars = list(DataFrame.columns)
    cols_vars.remove(target)

    x_train = DataFrame.loc[train_index, cols_vars]
    y_train = DataFrame.loc[train_index, target]

    x_test = DataFrame.loc[test_index, cols_vars]
    y_test = DataFrame.loc[test_index, target]

    # Standard Scaler
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Store the params of scaler to export, if it turns a production model
    
    # Model: Linear Regression
    regr = LinearRegression()

    # Parameters
    regr.fit_intercept = True
    regr.positive = False
    
    # Applying models
    regr.fit(X=x_train, y=y_train)

    # Train Score

    # Predictions
    y_pred = regr.predict(X=x_test)

    # Test Score: R2, Pearson, MAE, RMSE, Bland-Altman
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Outout
    hyperparams, params, results = dict(), dict(), dict()

    results["R2 Score"] = r2
    results["MAE"] = mae

    return hyperparams, params, results
    

# Program --------------------------------------------------------------
df = load_dataset()
target = "compressive_strength_mpa"

# Train/Validation/Test strategy
trainval, test = split_train_test(df, train_size=70, seed=314)
skf_folds = stratified_continuous_kfold(trainval, target=target) 
#           (fold_no, train_index, validation_index)

for (i, train_index, test_index) in skf_folds:
    _, _, results = prep_pipeline(df, train_index, test_index, target)
    print(results)


# end
