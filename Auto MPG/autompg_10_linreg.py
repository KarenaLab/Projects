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
df = df.drop(columns=["model_year", "origin", "car_name"])

# Data preparation
x, y = target_split(df, target="kpl")

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=137)

x_train, x_test = scaler(x_train, x_test, method=StandardScaler())

_, results = regr_linregr(x_train, x_test, y_train, y_test)

print(results)

# end

