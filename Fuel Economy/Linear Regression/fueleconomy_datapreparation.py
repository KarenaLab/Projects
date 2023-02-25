
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
from data_preparation import *
from plot_histogram import *
from plot_missingdata_col import *
from plot_missingdata_row import *
from plot_heatmap import *
from plot_scatterhist import *


# Setup/Config
seed = 27
np.random.seed(seed)

path_main = os.getcwd()
path_database = os.path.join(path_main, "Database")
path_report = os.path.join(path_main, "Report")


# Functions
def read_csv(path, filename, sep=",", encoding="utf-8"):
    """
    Internal function for reading database from folder/filename.

    """
    path_origin = os.getcwd()

    os.chdir(path)   
    data = pd.read_csv(filename, sep=sep, encoding=encoding, low_memory=False)
    os.chdir(path_origin)

    return data
   

# Program --------------------------------------------------------------
print(" ****  Fuel Economy  **** \n")

df = read_csv(path_database, filename="fueleconomy_us84.csv")

"""
# Standardize Columns name
df = column_prep(df)

# Filtering by year
year_min = 2010
year_max = 2011
df = df[(df["year"] >= year_min) & (df["year"] <= year_max)]

# Data Analysis: NaN Count
df = nan_count(df)
"""
