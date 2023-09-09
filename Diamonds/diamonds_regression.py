# Diamonds project [P063]
# Diamonds project learning


# Libraries
import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# Personal modules
sys.path.append(r"c:\python_modules")

from dataframe_preparation import *


# Functions
def rename_size_columns(DataFrame):
    """
    Rename size related columns (x, y, and z) to avoid problems with
    naming.

    """
    cols_rename = {"x": "size_x",
                   "y": "size_y",
                   "z": "size_z"}

    DataFrame = DataFrame.rename(columns=cols_rename)


    return DataFrame

# Setup/Config



# Program --------------------------------------------------------------

# DataFrame preparation
filename = "diamonds.csv"
df = read_csv(filename)

df = rename_size_columns(df)



# end
