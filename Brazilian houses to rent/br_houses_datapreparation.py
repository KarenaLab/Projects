# Brazilian houses to rent [P403]

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
    filename = "br_houses_to_rent_v1.csv"
    data = pd.read_csv(filename, index_col=0, sep=",", encoding="utf-8")
    data = data_preparation(data)

    return data


def data_preparation(DataFrame):

    def remove_moneytag(Series):
        data = Series.copy()

        data = data.str.replace("R$", "", regex=False)
        data = data.str.replace(",", "", regex=False)
        data = pd.to_numeric(data, errors="coerce")

        return data
    
    # Data preparation
    data = DataFrame.copy()

    # Columns names
    cols_dict = dict()
    for old_name in data.columns:
        new_name = old_name.lower()
        new_name = new_name.replace(" ", "_")\
                           .replace("-", "_")\
                           .replace(":", "")

        cols_dict[old_name] = new_name

    data = data.rename(columns=cols_dict)

    # Numeric columns as string
    data["floor"] = pd.to_numeric(data["floor"], errors="coerce")

    cols_money = ["hoa", "rent_amount", "property_tax", "fire_insurance",
                  "total"]

    for col in cols_money:
        data[col] = remove_moneytag(data[col])


    return data

    
# Setup/Config



# Program --------------------------------------------------------------
df = load_dataset()



# end

