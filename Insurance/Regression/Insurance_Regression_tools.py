
# Libraries
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Personal modules
sys.path.append(r"C:\python_modules")

from color_source import *
from plot_histogram import *
from plot_heatmap import *
from one_hot_encoding import *
from split_trainval_test import *


# Setup/Config
path_main = os.getcwd()
path_report = os.path.join(path_main, "Report")




# Functions
cols_numeric = ["age", "bmi", "charges"]
cols_categoric = ["sex", "children", "smoker", "region"]    


# Program --------------------------------------------------------------------
filename = "insurance.csv"
df = pd.read_csv(filename, sep=",", encoding="utf-8")
df = one_hot_encoding(df, columns=cols_categoric)

df_trainval, df_test = split_trainval_test(df, train_size=.8, seed=18)











