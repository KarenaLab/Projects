
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


# Setup/Config
path_main = os.getcwd()
path_report = os.path.join(path_main, "Report")




# Functions
cols_numeric = ["age", "bmi", "charges"]
cols_categoric = ["sex", "children", "smoker", "region"]

def split_trainval_test(DataFrame, trainval_size=80, seed=None):
    """


    """
    data = DataFrame.copy()

    if(seed != None):
        np.random.seed(seed)

    if(trainval_size >= 1):
        trainval_size = trainval_size / 100

    n_cut = int(data.shape[0] * trainval_size)

    # Shuffling entire dataset
    data = data.sample(frac=1)

    data_trainval = data.iloc[0:n_cut, :].reset_index(drop=True)
    data_test = data.iloc[n_cut: , :].reset_index(drop=True)
         


    return data_trainval, data_test
    


# Program --------------------------------------------------------------------
filename = "insurance.csv"
df = pd.read_csv(filename, sep=",", encoding="utf-8")
df = one_hot_encoding(df, columns=cols_categoric)

df_trainval, df_test = split_trainval_test(df, trainval_size=.8, seed=18)











