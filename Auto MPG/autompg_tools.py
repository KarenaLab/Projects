# AutoMPG [P316]


# Insights, improvements and bugfix
#


# Libraries
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.stats as st

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt



# ----------------------------------------------------------------------
def load_dataset():
    """
    Load and prepare **AutoMPG** dataset for further analysis and
    modeling.

    """
    filename = "auto_mpg.csv"
    data = pd.read_csv(filename, sep=",", encoding="utf-8")
    data = data_preparation(data)

    return data


def data_preparation(DataFrame, clean_dataframe=True):
    """
    Prepares AutoMPG dataframe.

    """
    data = DataFrame.copy()

    # Columns names
    col_names = dict()
    for old_name in data.columns:
        new_name = old_name.lower()
        new_name = new_name.replace(" ", "_")\
                           .replace("-", "_")

        col_names[old_name] = new_name

    data = data.rename(columns=col_names)

    # Units
    data["kpl"] = data["mpg"] * 0.42514371
    data = data.drop(columns=["mpg"])

    data["displacement_cm3"] = data["displacement"] * 16.387064
    data = data.drop(columns=["displacement"])

    data = data.rename(columns={"horsepower": "power_hp"})

    data["weight_kg"] = data["weight"] * 0.45359237
    data = data.drop(columns=["weight"])

    # Numeric
    data["power_hp"] = pd.to_numeric(data["power_hp"], errors="coerce")

    # Clean DataFrame
    if(clean_dataframe == True):
        data = data.dropna()
        data = data.drop_duplicates()
        data = data.reset_index(drop=True)
    

    return data


def target_split(DataFrame, target):
    """
    Splits a dataframe into variables and target.

    """
    x = DataFrame.drop(columns=[target])
    y = DataFrame[target]

    return x, y


def scaler(x_train, x_test, method=StandardScaler()):
    """
    Applies the **method** (that should be a scikit-learn function)
    to x_train and x_test.

    """
    sc = method
    sc = sc.fit(x_train)

    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)

    return x_train, x_test


def regr_linregr(x_train, x_test, y_train, y_test, fit_intercept=True,
                 positive=False):
    """
    Applies a **Linear Regression** model

    """
    # Data preparation
    params = namedtuple("parameters", ["intercept", "coefs"])

    # Model
    regr = LinearRegression()

    # Hyperparameters
    regr.fit_intercept = fit_intercept
    regr.positive = positive

    # Fit and predict
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    parameters = params(regr.intercept_, regr.coef_)

    # Metrics
    results = regr_metrics(y_test, y_pred)


    return params, results


def regr_metrics(y_true, y_pred):
    """
    Calculates the metrics regression.
    
    """
    # Data preparation
    metrics = namedtuple("results", ["mae", "rmse", "r2", "pearson"])

    # Calc
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pearson = st.pearsonr(y_true, y_pred).statistic

    results = metrics(mae, rmse, r2, pearson)

    return results
    
