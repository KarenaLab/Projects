
# Libraries
import os

import numpy as np
import pandas as pd



# Personal modules
import sys
sys.path.append(r"C:\python_modules")

from autoindustry_tools import *

from data_preparation import *
from split_kfold import *
from scalers_builtin import *
from regr_linreg import *

from plot_histogram import *
from plot_scatterhistlinreg import *
from plot_boxplot import *


# Functions


# Setup/Config
seed = 302
target = "power_hp"
n_splits = 5



# Program --------------------------------------------------------------
filename = "auto_industry.csv"
df = read_csv(filename)


# First Model, very simple (without origin and name)
cols_remove = ["origin", "name", "model_year"]
df = df.drop(columns=cols_remove)
df = dataframe_preparation(df)
df = units_conversion(df)

# Featuring Engineering
df["weight_power_ratio"] = df["weight_kg"] / df["power_hp"]


# Target split
x, y = split_target(df, target=target)


# Train/Test split
np.random.seed(seed)
split_table = split_kfold(x, n_splits=n_splits)


# Model Base = Linear Regression
for i in range(0, n_splits):
    fold, train_index, test_index = split_table[i]
    



