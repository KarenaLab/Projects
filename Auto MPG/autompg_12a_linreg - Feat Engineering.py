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
from plot_histbox import *


# Program --------------------------------------------------------------
df = load_dataset()
df = df.drop(columns=["model_year", "origin", "car_name"])
df["power_to_weight"] = df["power_hp"] / df["weight_kg"]

# Data preparation
x, y = target_split(df, target="kpl")

df_results = pd.DataFrame(data=[])
np.random.seed(137)

for seed in np.random.randint(low=0, high=1000, size=50):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=seed)
    x_train, x_test = scaler(x_train, x_test, method=StandardScaler())
    _, results = regr_linregr(x_train, x_test, y_train, y_test)

    df_results = store_results(df_results, results)


for col in df_results.columns:
    plot_histbox(df_results[col], title=f"AutoMPG - 12 - LinRegr with Feat Eng - {col}", savefig=True)

    mean = df_results[col].mean()
    stddev = df_results[col].std()
    print(f" > {col}: {mean:.4f} +/- {stddev:.4f}")
    

# end
