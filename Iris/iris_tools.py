# Iris - Project [P336]
# Learning machine learning fundamentals with Iris project

# Libraries
import numpy as np
import pandas as pd

import scipy.stats as st

import matplotlib.pyplot as plt



# ----------------------------------------------------------------------
def load_dataset():
    """
    Extract, transform and load data from Iris dataset and prepares it
    for further analysis and processing.

    """
    # Data load
    filename = "iris.csv"
    data = pd.read_csv(filename, sep=",", encoding="utf-8")

    # Columns preparation
    col_name = dict()
    
    for old_name in data.columns:
        new_name = old_name.replace("length", "length_cm")\
                           .replace("width", "width_cm")

        col_name[old_name] = new_name


    data = data.rename(columns=col_name)
            

    return data


def target_split(DataFrame, target):
    """
    Splits **DataFrame** into **x** and **y**, 

    """
    x = DataFrame.drop(columns=[target])
    y = DataFrame[target]

    return x, y



