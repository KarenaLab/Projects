# AutoMPG [P316]

# Libraries
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as st

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# Personal modules
sys.path.append(r"c:\python_modules")

from autompg_tools import *
from plot_lineduo import plot_lineduo


# Program --------------------------------------------------------------
df = load_dataset()
df = df.drop(columns=["car_name"])
df = df.drop(columns=["model_year", "origin"])

# List of indexes shuffled
indexes = np.array(df.index)
np.random.seed(137)
np.random.shuffle(indexes)

# Variables and target split
target = "kpl"
x_vars = df.columns.to_list()
x_vars.remove(target)

train_size_list = np.linspace(start=5, stop=95, num=18+1)

inside_pearson = list()
outside_pearson = list()

for i in train_size_list:
    train_size = int((indexes.size * (i/100)) + 0.5)

    train_index = indexes[0:train_size]
    test_index = indexes[train_size: ]

    x_train = df.loc[train_index, x_vars]
    x_test = df.loc[test_index, x_vars]
    y_train = df.loc[train_index, target]
    y_test = df.loc[test_index, target]

    x_train, x_test = scaler(x_train, x_test, method=StandardScaler())

    # Model
    regr = LinearRegression()

    # Hyperparameters
    regr.fit_intercept = True
    regr.positive = False

    # Fit and intercept
    regr.fit(x_train, y_train)
    y_inside = regr.predict(x_train)
    y_outside = regr.predict(x_test)

    # Metrics
    res_inside = regr_metrics(y_train, y_inside)
    res_outside = regr_metrics(y_test, y_outside)

    inside_pearson.append(res_inside["pearson"])
    outside_pearson.append(res_outside["pearson"])


plot_lineduo(x1=train_size_list, y1=inside_pearson, label1="train",
             y2=outside_pearson, label2="test",
             title=f"AutoMPG - 10b - Train and test performance",
             xlabel="train size (%)", ylabel="pearson", legend_loc="lower left",
             savefig=True)


# end
