# FIPE Cars [P401]

# Libraries
import os
import sys

import numpy as np
import pandas as pd

import scipy.stats as st

import matplotlib.pyplot as plt


# Personal modules
sys.path.append(r"c:\python_modules")


# Functions
def load_dataset():
    filename = "fipe_cars_brasil_2022.csv"
    data = pd.read_csv(filename, index_col=0, sep=",", encoding="utf-8")
    data = data_preparation(data)

    return data


def data_preparation(DataFrame):
    data = DataFrame.copy()

    cols_dict = dict()
    for old_name in data.columns:
        new_name = old_name.lower()
        new_name = new_name.replace(" ", "_")\
                           .replace("-", "_")

        cols_dict[old_name] = new_name

    data = data.rename(columns=cols_dict)

    data["marca"] = data["marca"].str.lower()


    return data

    


# Setup/Config



# Program --------------------------------------------------------------
df = load_dataset()



# end

