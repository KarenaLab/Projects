
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

from plot_barv import plot_barv
from plot_histbox import plot_histbox
from plot_scatterhist import plot_scatterhist
from bins_calculator import bins_alloptions


# Functions
def load_dataset():
    """
    Load dataset for FIEC Test (Canada Retail Profit Case).

    """
    # Extract: Import data (raw)
    filename = "dados.csv"
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


def split_target(DataFrame, target):
    x = DataFrame.drop(columns=[target])
    y = DataFrame[[target]]

    return x, y


def regr_metrics(y_true, y_pred):
    """


    """
    # Data preparation
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearsonr = st.pearsonr(y_true, y_pred).statistic

    results = dict()
    results["mae"]= mae
    results["rmse"] = rmse
    results["r2_score"] = r2
    results["pearsonr"] = pearsonr

    return results
    

# Program
df = load_dataset()
"""
for col in cat_cols():
    data = df[col].value_counts()
    x = np.array(data.index)
    y = list(data.values)
    plot_barv(x, height=y, title=f"ecommerce - Category - {col}", savefig=True)
    
for col in num_cols():
    data = df[col]
    plot_histbox(data, title=f"ecommerce - Histogram - {col}", savefig=True)

"""

# Numerical to Categoric
df["sales_cat"] = numeric_to_categoric(df["sales"], no_cat=10)
df = df.drop(columns=["sales"])

df["order_quantity_cat"] = numeric_to_categoric(df["order_quantity"], no_cat=10)
df = df.drop(columns=["order_quantity"])

# Train/Test split
x, y = split_target(df, target="profit")
x = pd.get_dummies(x, drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Model
regr = DecisionTreeRegressor()

# Hyperparameters


# Fit and predict
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
metrics = regr_metrics(y_test, y_pred)
print(metrics)




















