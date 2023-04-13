
# Libraries
import os

from itertools import product

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

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
    new_values = []
    for i in range(0, data.shape[0]):
        value = data.loc[i, "horsepower"]

        if(value.isdigit() == True):
            value = float(value)

        else:
            value = np.nan

        new_values.append(value)

    data["horsepower"] = new_values

        
    return data


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


# Program --------------------------------------------------------------
print("\n ****  Auto MPG Machine Learning  **** \n")

# Import Dataset
filename = "auto_mpg.csv"
df = read_csv(filename)
df = convert_to_SI(df)

# Remove **origin** and **car_name**
df = df.drop(columns=["origin", "car_name"])

nan_index = df.loc[pd.isna(df["horsepower"]), :].index

if(nan_index.size > 0):
    df_solve = df.loc[nan_index, :].reset_index(drop=True)
    df = df.drop(index=nan_index).reset_index(drop=True)


# Regression = Base Line
target = "kpl"
x = df.drop(columns=[target])
y = df[target]

# Data Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state=3)

# Scaler
scaler = StandardScaler()
scaler = scaler.fit(x_train)

x_train_sc = scaler.transform(x_train)
x_test_sc = scaler.transform(x_test)


# Modeling
results = pd.DataFrame(data=[],
                       columns=["name", "fit_intercept", "positive", "alpha", "l1_ratio", "mae", "rmse", "r2"])

# Model 01: Linear Regression (Model Base)
model = LinearRegression()

# Model Hiperparameters
fit_intercept = [True, False]
positive = [False, True]

model_params = list(product(fit_intercept, positive))

for fi, p in model_params:
    # Setting hiperparameters
    model.fit_intercept = fi
    model.positive = p

    model_info = {"name": type(model).__name__,
                  "fit_intercept": fi,
                  "positive": p}

    # Model apply    
    model.fit(x_train_sc, y_train)
    y_pred = model.predict(x_test_sc)

    model_metrics = metrics(y_test, y_pred)
    model_info.update(model_metrics)

    _values = pd.Series(model_info)
    results = pd.concat([results, _values.to_frame().T], ignore_index=True)


# Models 02 and 03: Ridge and Lasso
# Model Hiperparameters
model_list = [Ridge(), Lasso()]

fit_intercept = [True, False]
positive = [False, True]
alpha = [1.0000e-03, 3.1623e-02, 1.0000e-02, 3.1623e-01,
         1.0000e-01, 3.1623e+00, 1.0000e+00, 3.1623e+01,
         1.0000e+01, 3.1623e+02, 1.0000e+02, 3.1623e+03,
         1.0000e+03]

model_params = list(product(fit_intercept, positive, alpha))

for m in model_list:
    # Setting Model
    model = m
    
    for fi, p, a in model_params: 
        # Setting hiperparameters
        model.fit_intercept = fi
        model.positive = p
        model.alpha = a

        model_info = {"name": type(model).__name__,
                      "fit_intercept": fi,
                      "positive": p,
                      "alpha": a}

        # Model apply    
        model.fit(x_train_sc, y_train)
        y_pred = model.predict(x_test_sc)

        model_metrics = metrics(y_test, y_pred)
        model_info.update(model_metrics)

        _values = pd.Series(model_info)
        results = pd.concat([results, _values.to_frame().T], ignore_index=True)


# Model 04: ElasticNet
model = ElasticNet()

fit_intercept = [True, False]
positive = [False, True]
alpha = [1.0000e-03, 3.1623e-02, 1.0000e-02, 3.1623e-01,
         1.0000e-01, 3.1623e+00, 1.0000e+00, 3.1623e+01,
         1.0000e+01, 3.1623e+02, 1.0000e+02, 3.1623e+03,
         1.0000e+03]
l1_ratio = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
            0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99, 1.00]

model_params = list(product(fit_intercept, positive, alpha, l1_ratio))

for fi, p, a, l1 in model_params: 
    # Setting hiperparameters
    model.fit_intercept = fi
    model.positive = p
    model.alpha = a
    model.l1_ratio = l1

    model_info = {"name": type(model).__name__,
                  "fit_intercept": fi,
                  "positive": p,
                  "alpha": a,
                  "l1_ratio": l1}

    # Model apply    
    model.fit(x_train_sc, y_train)
    y_pred = model.predict(x_test_sc)

    model_metrics = metrics(y_test, y_pred)
    model_info.update(model_metrics)

    _values = pd.Series(model_info)
    results = pd.concat([results, _values.to_frame().T], ignore_index=True)



# end
