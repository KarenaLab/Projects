
# Libraries
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

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
df_solve = df.loc[nan_index, :].reset_index(drop=True)
df = df.drop(index=nan_index).reset_index(drop=True)


# Histogram
categoric_dict = {"cylinders": ["cylinders", "No. cylinders"],
                  "model_year": ["model_year", "model year (yr)"]}

for i in categoric_dict.keys():
    col, xlabel = categoric_dict[i]

    title = f"AutoMPG - Histogram {col}"
    plot_histogram(df[col], title=title, xlabel=xlabel, bins="sqrt",
                   kde=False, meanline=False, savefig=savefig)


continuous_dict = {"displacement": ["displacement", "engine displacement (cm^3)"],
                   "horsepower": ["horsepower", "engine horsepower (hp)"],
                   "weight": ["weight", "weight (kg)"],
                   "acceleration": ["acceleration", "acceleration (0-96 kph)"],
                   "kpl": ["kpl", "consumption (km per liter)"]}


for i in continuous_dict.keys():
    col, xlabel = continuous_dict[i]

    title = f"AutoMPG - Histogram {col}"
    plot_histogram(df[col], title=title, xlabel=xlabel, bins="sqrt",
                   kde=True, meanline=True, savefig=savefig)


# end
