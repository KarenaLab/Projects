
# Libraries
import os

from itertools import product

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt


# Personal modules
import sys
sys.path.append(r"c:\python_modules")

from file_management import *
from plot_histogram import *


# Setup/Config
path_script = os.getcwd()
path_report = os.path.join(path_script, "report")

savefig = False


# Functions
def read_csv(filename, sep=",", encoding="utf-8"):
    """
    Built-in function for Auto mpg dataset reading.

    """
    data = pd.read_csv(filename, sep=sep, encoding=encoding)

    for col in data.columns:
        new_col = col[:]
        new_col = new_col.lower()\
                         .replace(" ", "_")

        data = data.rename(columns={col: new_col})


    return data


def convert_to_SI(DataFrame):
    """
    Converts the dataset in Imperial units to International system units.
    It will not change the final results but will make it easier to under-
    stand the values.

    """
    data = DataFrame.copy()

    # Consumption: mpg to km/l (target)
    data["kpl"] = data["mpg"].apply(lambda x: x * 0.42514371)
    data = data.drop(columns=["mpg"])

    # Displacement: Cubic Inches (in^3) to liters (cm^3)
    data["displacement"] = data["displacement"].apply(lambda x: np.round_(x * 0.016387064, decimals=5))

    # Weight: Pound to kg
    data["weight"] = data["weight"].apply(lambda x: np.round(x * 0.45359237, decimals=2))
    
    # Horsepower: text to float
    data["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")

        
    return data


def KFold_split(DataFrame, n_splits=5, random_state=None):
    """


    """
    if(random_state == None):
        kf = KFold(n_splits=n_splits)

    else:
        kf = KFold(n_splits=n_splits,
                   random_state=random_state, shuffle=True)


    test_index_list = []
    for _, test in kf.split(DataFrame):
        test_index_list.append(test)


    return test_index_list

def folders_split(DataFrame, target, test_index):
    """


    """
    var_list = DataFrame.columns.to_list()
    var_list.remove(target)

    x_test = DataFrame.loc[test_index, var_list].reset_index(drop=True)
    y_test = DataFrame.loc[test_index, target].reset_index(drop=True)

    x_train = DataFrame.drop(index=test_index, columns=target).reset_index(drop=True)
    y_train = DataFrame.drop(index=test_index, columns=var_list).reset_index(drop=True)


    return x_train, y_train, x_test, y_test


def model_linearregression(x_train, y_train, x_test, y_test,
                           fit_intercept=True, positive=False):
    """


    """
    model = LinearRegression()

    model.fit_intercept = fit_intercept
    model.positive = positive

    model_info = {"name": "Linear Regression",
                  "fit_intercept": fit_intercept,
                  "positive": positive}

    # Model apply    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    model_metrics = metrics(y_test, y_pred)
    model_info.update(model_metrics)


    return model_info


def model_ridge(x_train, y_train, x_test, y_test,
                alpha=1, fit_intercept=True, positive=False):
    """


    """
    model = Ridge()

    model.alpha = alpha
    model.fit_intercept = fit_intercept
    model.positive = positive
    
    model_info = {"name": "Ridge",
                  "alpha": alpha,
                  "fit_intercept": fit_intercept,
                  "positive": positive}

    # Model apply    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    model_metrics = metrics(y_test, y_pred)
    model_info.update(model_metrics)


    return model_info


def model_lasso(x_train, y_train, x_test, y_test,
                alpha=1, fit_intercept=True, positive=False):
    """


    """
    model = Lasso()

    model.alpha = alpha
    model.fit_intercept = fit_intercept
    model.positive = positive
    
    model_info = {"name": "Lasso",
                  "alpha": alpha,
                  "fit_intercept": fit_intercept,
                  "positive": positive}

    # Model apply    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    model_metrics = metrics(y_test, y_pred)
    model_info.update(model_metrics)


    return model_info


def model_elasticnet(x_train, y_train, x_test, y_test,
                     alpha=1, l1_ratio=0.5 ,fit_intercept=True, positive=False):
    """


    """
    model = Lasso()

    model.alpha = alpha
    model.l1_ratio = l1_ratio
    model.fit_intercept = fit_intercept
    model.positive = positive
    
    
    model_info = {"name": "Lasso",
                  "alpha": alpha,
                  "l1_ratio": l1_ratio,
                  "fit_intercept": fit_intercept,
                  "positive": positive}

    # Model apply    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    model_metrics = metrics(y_test, y_pred)
    model_info.update(model_metrics)


    return model_info


def metrics(y_true, y_pred, decimals=4):
    """
    Returns Regression metrics (MAE, RMSE and R2 Score) for the y_true
    (or Ground Truth) and y_pred (Estimated).
    
    """
    mae = np.round(mean_absolute_error(y_test, y_pred), decimals=decimals)
    rmse = np.round(mean_squared_error(y_test, y_pred), decimals=decimals)
    r2 = np.round(r2_score(y_test, y_pred), decimals=decimals)

    results = {"mae": mae, "rmse": rmse, "r2": r2}

    return results


def store_results(DataFrame, new_row):
    """


    """
    new_row = pd.Series(new_row)
    DataFrame = pd.concat([DataFrame, new_row.to_frame().T], ignore_index=True)


    return DataFrame



# Program --------------------------------------------------------------
print("\n ****  Auto MPG Machine Learning  **** \n")

# Import Dataset
filename = "auto_mpg.csv"
df = read_csv(filename)
df = convert_to_SI(df)

# Remove **origin** and **car_name**
df = df.drop(columns=["origin", "car_name"])


# Feature Engineering
df["weight_power_ratio"] = df["weight"] / df["horsepower"]
df["power_per_volume"] = df["horsepower"] / df["displacement"]


# Separate values with np.nan (further object of analysis)
nan_index = df.loc[pd.isna(df["horsepower"]), :].index

if(nan_index.size > 0):
    df_solve = df.loc[nan_index, :].reset_index(drop=True)
    df = df.drop(index=nan_index).reset_index(drop=True)


# Modeling
results_cols = ["name", "folder", 
                "fit_intercept", "positive", "alpha", "l1_ratio",
                "mae", "rmse", "r2"]


results = pd.DataFrame(data=[], columns=results_cols)

# KFold Data Split
test_indexes = KFold_split(df, n_splits=5, random_state=3)
target = "kpl"

# Model 01: Linear Regression (Model Base)
folds = range(0, len(test_indexes))
fit_intercept = [True, False]
positive = [False, True]

for fi, p, i in list(product(fit_intercept, positive, folds)):
    x_train, y_train, x_test, y_test = folders_split(df, target, test_indexes[i])

    # Scaler
    scaler = StandardScaler()
    scaler = scaler.fit(x_train)

    x_train_sc = scaler.transform(x_train)
    x_test_sc = scaler.transform(x_test)

    info = model_linearregression(x_train_sc, y_train, x_test_sc, y_test,
                                  fit_intercept=fi, positive=p)
    info["folder"] = i

    results = store_results(results, info)


# Model 02: Ridge 
# Model Hiperparameters
alpha = [1.0000e-03, 3.1623e-02, 1.0000e-02, 3.1623e-01,
         1.0000e-01, 3.1623e+00, 1.0000e+00, 3.1623e+01,
         1.0000e+01, 3.1623e+02, 1.0000e+02, 3.1623e+03,
         1.0000e+03]

for fi, p, a, i in list(product(fit_intercept, positive, alpha, folds)):
    x_train, y_train, x_test, y_test = folders_split(df, target, test_indexes[i])

    # Scaler
    scaler = StandardScaler()
    scaler = scaler.fit(x_train)

    x_train_sc = scaler.transform(x_train)
    x_test_sc = scaler.transform(x_test)

    info = model_ridge(x_train_sc, y_train, x_test_sc, y_test,
                       fit_intercept=fi, positive=p, alpha=a)
    info["folder"] = i

    results = store_results(results, info)


# Models 03: Lasso 
# Attention: Using the same parameters from Ridge.

for fi, p, a, i in list(product(fit_intercept, positive, alpha, folds)):
    x_train, y_train, x_test, y_test = folders_split(df, target, test_indexes[i])

    # Scaler
    scaler = StandardScaler()
    scaler = scaler.fit(x_train)

    x_train_sc = scaler.transform(x_train)
    x_test_sc = scaler.transform(x_test)

    info = model_lasso(x_train_sc, y_train, x_test_sc, y_test,
                       fit_intercept=fi, positive=p, alpha=a)
    info["folder"] = i

    results = store_results(results, info)


# Model 04: ElasticNet
# Model Hiperparameters
l1_ratio = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
            0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99, 1.00]

for fi, p, a, l1, i in list(product(fit_intercept, positive, alpha, l1_ratio, folds)):
    x_train, y_train, x_test, y_test = folders_split(df, target, test_indexes[i])

    # Scaler
    scaler = StandardScaler()
    scaler = scaler.fit(x_train)

    x_train_sc = scaler.transform(x_train)
    x_test_sc = scaler.transform(x_test)

    info = model_elasticnet(x_train_sc, y_train, x_test_sc, y_test,
                            fit_intercept=fi, positive=p, alpha=a, l1_ratio=l1)
    info["folder"] = i

    results = store_results(results, info)


# Preparing Results dataset
cols_list = ["folder", "alpha", "l1_ratio", "mae", "rmse", "r2"]

for col in cols_list:
    results[col] = results[col].astype("float")
    
    
# end
