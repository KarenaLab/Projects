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


# Program --------------------------------------------------------------
df = load_dataset()
df = df.drop(columns=["car_name"])
df = df.drop(columns=["model_year", "origin"])

# Data preparation
x, y = target_split(df, target="kpl")

seed = 147
train_size = 0.8

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=seed)
x_train, x_test = scaler(x_train, x_test, method=StandardScaler())
_, results = regr_linregr(x_train, x_test, y_train, y_test)

print(results)


# First round
# train_size:0.8, seed:331 - pearson: 0.87212
# train_size:0.8, seed:734 - pearson: 0.83041
# train_size:0.8, seed:439 - pearson: 0.81095

# Second round (after train-test performance graph)
# train_size:0.65, seed:589 - pearson: 0.83752
# train_size:0.65, seed:294 - pearson: 0.81796
# train_size:0.65, seed:240 - pearson: 0.83126

# (( values could change with python, numpy and scikit-learn versions ))

# end
