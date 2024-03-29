# AutoMPG [P316]

# Libraries
import os
import sys
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.stats as st

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# Personal modules
sys.path.append(r"c:\python_modules")

from autompg_tools import *
from plot_line import *


# Program --------------------------------------------------------------
df = load_dataset()
df = df.drop(columns=["car_name"])
df = df.drop(columns=["model_year", "origin"])
df["power_to_weight"] = df["power_hp"] / df["weight_kg"]

# Data preparation
kf_folds = kfold_generate(df, shuffle=True, random_state=137)
alpha_list = np.linspace(start=0.001, stop=0.02, num=100)

df_results = pd.DataFrame(data=[])
for fold, train_index, test_index in kf_folds:
    x_train, x_test, y_train, y_test = fold_split(df, train_index, test_index, target="kpl")
    x_train, x_test = scaler(x_train, x_test, method=StandardScaler())

    for alpha in alpha_list:
        _, results = regr_lasso(x_train, x_test, y_train, y_test, alpha=alpha)
        results["alpha"] = alpha
        results["fold"] = fold

        df_results = store_results(df_results, results)


# Organize
df_results = df_results[["alpha", "fold", "pearson", "rmse", "mae"]]
df_results = df_results.sort_values(by=["alpha", "fold"], ignore_index=True)

mae_list = list()
rmse_list = list()
pearson_list = list()

# Sumarize folds performance by alpha
for alpha in alpha_list:
    data = df_results.groupby(by="alpha").get_group(alpha)

    mae_list.append(data["mae"].mean())
    rmse_list.append(data["rmse"].mean())
    pearson_list.append(data["pearson"].mean())

# Plots
for ylabel, metric in zip(["MAE", "RMSE", "Pearson R"], [mae_list, rmse_list, pearson_list]):
    plot_line(alpha_list, metric, title=f"AutoMPG - 16 - Lasso - {ylabel}",
              xlabel="alpha param", ylabel=ylabel, savefig=False)
    

# end
