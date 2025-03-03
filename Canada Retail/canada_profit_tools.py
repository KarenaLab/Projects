
# Libraries
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as st

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

# Personal modules
sys.path.append(r"c:\python_modules")



# Functions
def load_dataset():
    """
    Load dataset for FIEC Test (Canada Retail Profit Case).

    """
    # Extract: Import data (raw)
    filename = "canada_retail.csv"
    data = pd.read_csv(filename, sep=",", encoding="utf-8")

    # Transform: Prepare data
    data = data_preparation(data)

    
    return data


def data_preparation(DataFrame):
    """
    Transforms data from extraction

    """
    data = DataFrame.copy()

    # Rename columns (standard lower and remove (-) sign
    cols_name = dict()
    for name in data.columns:
        new_name = name.lower()
        new_name = new_name.replace("-", "")
        cols_name[name] = new_name

    data = data.rename(columns=cols_name)

    # Remove `order_id` and `customer_name` (Data Privacy)
    data = data.drop(columns=["order_id", "customer_name"])
    

    return data


def cat_cols():
    cols = ["order_priority",  "ship_mode", "region", "customer_segment",
            "product_category", "product_subcategory", "product_container"]

    return cols


def num_cols():
    cols = ["order_quantity", "sales", "profit"]

    return cols


def text_cols():
    cols = ["product_name"]

    return cols


def numeric_to_categoric(data, no_cat=5):
    # Data preparation
    data = np.array(data)

    # Histogram based ;)
    _, bins_edges = np.histogram(data, bins=no_cat)

    # Segment classification
    segment_list = np.array([])

    for value in data:
        segment = np.sum(value > bins_edges)

        if(segment == 0):
            # Solving the problem of creating only one bin with
            #     number 0. Colapsing it with the first bin (n=1).
            #     It will work only one time.
            segment = 1

        segment_list = np.append(segment_list, f"c{segment}")


    return segment_list    

