
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
from print_results import *

from plot_histogram import *
from plot_scatterhistlinreg import *
from plot_boxplot import *
from plot_identityline import *
from plot_blandaltman import *


# Functions
def append_results(DataFrame, new_dict):
    """
    Append a dictionary with **new_row_dict** to a **DataFrame**
    
    """
    DataFrame = pd.concat([DataFrame, pd.Series(new_dict).to_frame().T],
                          ignore_index=True)

    return DataFrame


def flatten_list(array):
    """
    Turns a list of lists into a flat list.
    
    """
    flatten = list()
    
    for sub_list in array:
        for i in sub_list:
            flatten.append(i)


    return flatten


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


results = results_to_dataframe(flatten_list(results))







    

