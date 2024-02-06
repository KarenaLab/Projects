# AutoMPG [P316]
# (optional) Short description of the program/module.


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


# Functions



# Setup/Config



# Program --------------------------------------------------------------
df = load_dataset()
df = df.drop(columns=["car_name"])
df = df.drop(columns=["model_year", "origin"])
df["power_to_weight"] = df["power_hp"] / df["weight_kg"]

# Data preparation
kf_folds = kfold_generate(df, shuffle=True, random_state=137)

df_results = pd.DataFrame(data=[])
for i, train_index, test_index in kf_folds:
    x_train, x_test, y_train, y_test = fold_split(df, train_index, test_index, target="kpl")
    x_train, x_test = scaler(x_train, x_test, method=StandardScaler())
    _, results = regr_linregr(x_train, x_test, y_train, y_test)
    df_results = store_results(df_results, results)

    print(results)
    

# end

