# Name [P127]
# Comparing the efficiency of a Bootstrap (from scratch) with a Linear
# Regression model.


# Libraries
import os
import sys

from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.stats as st

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt


# Personal modules
sys.path.append(r"c:\python_modules")


# Functions
def load_dataset(reset_index=True):
    """
    Load and prepares Height and Weight dataset.

    """
    # Upload
    filename = "kaggle_height_weight.csv"
    data  = pd.read_csv(filename, sep=",", encoding="utf-8")

    # Transforms for SI units (unit as sulfix)
    data["height_cm"] = np.round(data["Height"] * 2.54, decimals=5)
    data["weight_kg"] = np.round(data["Weight"] * 0.45359237, decimals=5)
    data = data.drop(columns=["Height", "Weight"])

    # Gender: Male=1, Female=0
    gender_dummy = {"Male": 1, "Female": 0}
    data = data.rename(columns={"Gender": "gender"})
    data["gender"] = data["gender"].map(gender_dummy)

    # Check for NaNs and duplicates
    data = data.dropna()
    data = data.drop_duplicates()

    if(reset_index == True):
        data = data.reset_index(drop=True)
    

    return data


def bootstrap_sorting(DataFrame, size=None):
    """


    """
    # Data preparation: Size
    if(isinstance(size, int) == True):
        pass

    else:
        size = DataFrame.shape[0]


    # Data preparation: Index numbers
    indexes = np.array(DataFrame.index)


    # Indexes for bag
    sorting = np.random.choice(indexes, size=size, replace=True, p=None)


    return sorting


def bagging(DataFrame, size=None):
    """


    """
    # Data preparation: Size
    if(isinstance(size, int) == True):
        pass

    else:
        size = DataFrame.shape[0]


    # Indexes for bag
    bag_index = bootstrap_sorting(DataFrame, size)
    bag_index = np.unique(bag_index)

    # Indexes for out-of-bag (OOB)
    oob_index = np.array(list(set(DataFrame.index) - set(bag_index)))


    return bag_index, oob_index


def regr_lasso():
    """


    """
    pass

    return None
    


    
        
    


# Setup/Config



# Program --------------------------------------------------------------
df = load_dataset()



# end
