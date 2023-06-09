
# Libraries
import os

import numpy as np
import pandas as pd


# Personal Modules
import sys
sys.path.append(r"C:\python_modules")

from data_preparation import *
from plot_histogram import *


# Setup/Config



# Program --------------------------------------------------------------
filename = "iris.csv"
df = pd.read_csv(filename, sep=",", encoding="utf-8")

df = col_preparation(df, method="lower")
df = remove_duplicates(df)
df = nan_counter(df)
_ = distinct_counter(df)

