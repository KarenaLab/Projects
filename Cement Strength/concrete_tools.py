# [P504] Concrete compressive strength


# Libraries
import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (r2_score, mean_absolute_error, root_mean_squared_error)

import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
def load_dataset():
    """
    Import dataset for **concrete_strength** project.

    """
    filename = "concrete_strength.csv"
    data = pd.read_csv(filename, sep=",", encoding="utf-8")

    # Remove duplicated
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    

    return data


def cols_variable():
    """
    Variables

    """
    cols = ["cement_kg_p_m3", "blast_furnace_slag_kg_p_m3",
            "fly_ash_kg_p_m3", "water_kg_p_m3", "superplasticizer_kg_p_m3",
            "coarse_aggregate_kg_p_m3", "fine_aggregate_kg_p_m3", "age_days"]

    return cols


def prep_pipeline(DataFrame, train_index, test_index, target):
    # Data split
    x_train, y_train, x_test, y_test = data_split(DataFrame, train_index, test_index, target)

    # Standard Scaler
    x_train, x_test = data_scaler(x_train, x_test)

    # Models
    # [1] Improve - Add hyperparams as args and kwargs
    hyperparams = dict()
    params = dict()
    results = dict()
    
    model_hp, model_p, model_r = linear_regression(x_train, y_train, x_test, y_test,
                                                   fit_intercept=True, positive=False)

    hyperparams = model_hp
    params = model_p
    results = model_r
   
    return hyperparams, params, results

    

def data_split(DataFrame, train_index, test_index, target):
    # Columns preparation
    variables = list(DataFrame.columns)
    variables.remove(target)

    # Data split
    x_train = DataFrame.loc[train_index, variables]
    y_train = DataFrame.loc[train_index, target]

    x_test = DataFrame.loc[test_index, variables]
    y_test = DataFrame.loc[test_index, target]

    return x_train, y_train, x_test, y_test


def data_scaler(x_train, x_test):
    scaler = StandardScaler()

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Prepare to returm the coeficients for inverse transform.

    return x_train, x_test
    

def linear_regression(x_train, y_train, x_test, y_test,
                      fit_intercept=True, positive=False):

    # Initialization
    regr = LinearRegression()

    # HyperParameters
    regr.fit_intercept = fit_intercept
    regr.positive = positive

    hyperparams = dict()
    hyperparams["model"] = type(regr).__name__
    hyperparams["fit_intercept"] = fit_intercept
    hyperparams["positive"] = positive

    # Applying model
    regr.fit(X=x_train, y=y_train)

    # Train score

    # Predictions and Parameters
    y_pred = regr.predict(x_test)

    params = dict()
    params["coefs"] = regr.coef_
    params["intercept"] = regr.intercept_

    # Metrics and Scores
    results = regr_metrics(y_test, y_pred)
    

    return hyperparams, params, results


def regr_metrics(y_true, y_pred):
    # Metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    corr_pearson, _ = stats.pearsonr(y_true, y_pred)

    # Results
    results = dict()
    results["R2 Score"] = r2
    results["MAE"] = mae
    results["RMSE"] = rmse
    results["Pearson"] = corr_pearson

    return results


def aggregate_results(DataFrame, results):
    # Index
    fold = results["fold"]
    
    for (var, value) in zip(results.keys(), results.values()):
        DataFrame.loc[fold, var] = value


    return DataFrame


