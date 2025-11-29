# Project Paper CO

# Libraries
import os
import itertools
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt


# Project libraries
from src.paper_co_tools import (load_dataset, organize_report)
from src.plot_histbox import plot_histbox
from src.plot_barv import plot_barv
from src.plot_scatterhist import plot_scatterhist
from src.plot_heatmap import plot_heatmap


# Functions
def check_failure_data(DataFrame):
    """
    Function to check IF the **asset_id** and **runtime** orchestration
    is working corretly. When runtime stops to increase, it means the machine
    had a failure and asset_id need to be increased.

    If working correctly, asset_id increase every time runtime resets for 1 (one).

    """
    # "asset_id" and "runtime".
    # **asset_id** representes a complete run of the asset until its failure,
    # after the failure, the asset_id value is increased.
    # **runtime** is a measure of time that resets after failure.
    #
    # So, every asset_id increase, runtime resets for 1.
    
    errors = list()
    index = list(DataFrame.index)[0:-1]

    for i in index:
        asset_t0 = df.loc[i, "asset_id"]
        runtime_t0 = df.loc[i, "runtime"]

        asset_t1 = df.loc[i+1, "asset_id"]
        runtime_t1 = df.loc[i+1, "runtime"]

        if(asset_t1 > asset_t0):
            if(runtime_t1 != 1):
                errors.append(asset_t0)

    return errors


def remove_cols_unique(DataFrame, verbose=False):
    """
    Remove columns with a single value, does not change during the
    data collection.

    Arguments:
    * DataFrame: Pandas DataFrame

    Return:
    * DataFrame: Processed Pandas DataFrame
    * verbose: True or False* (default=False)

    """
    cols_remove = list()
    for col in DataFrame.columns:
        if(DataFrame[col].nunique() == 1):
            cols_remove.append(col)

    if(len(cols_remove) > 0):
        DataFrame = DataFrame.drop(columns=cols_remove)

     # Verbose   
    if(verbose == True):
        for col in cols_remove:
            print(f" > Column '{col}' removed due unique item")


    return DataFrame


def feat_eng_runtime_inv(DataFrame):
    """
    Inverts the runtime cycle, number will be a count down to the failure.
    Important: Using **negative** numbers to avoid conflict with runtime values.

    Arguments:
    * DataFrame: Pandas DataFrame

    Return:
    * DataFrame: Processed Pandas DataFrame

    """
    
    for asset in DataFrame["asset_id"].unique():
        info = DataFrame.groupby(by="asset_id").get_group(asset)
        runtime_max = info["runtime"].max()
        info["runtime_inv"] = info["runtime"].apply(lambda x: x - runtime_max)

        for i in info.index:
            DataFrame.loc[i, "runtime_inv"] = info.loc[i, "runtime_inv"]


    return DataFrame


def plot_columns_nunique(DataFrame, title=None, savefig=False):
    """
    Function to check columns with a single/unique value.
    Data that could be important for process control but not relevant for
    machine learning models.

    Arguments:
    * DataFrame: Pandas DataFrame
    * title: Title for the plot and filename if savefig is enabled,
    * savefig: True or False* Selection to show or save the plot (default=False),

    Return:
    *** Shows or save a figure with data processing information.

    """
    # Data processing: Number of unique values per column
    cols_unique = pd.DataFrame(data=[])
    for col in df.columns:
        cols_unique.loc[col, "unique"] = df[col].nunique()

    # Plot
    plot_barv(x=cols_unique.index, height=cols_unique["unique"],
              title=title, xrotation=True, upside_down=False, savefig=savefig)

    return None

    
# Setup/Config ----------------------------------------------------------
path_main = os.getcwd()
path_database = os.path.join(path_main, "database")
path_report = os.path.join(path_main, "report")

SAVEFIG = True

warnings.filterwarnings("ignore")


# Program ---------------------------------------------------------------

# Import dataset for EDA
df = load_dataset(filename="pm_train.txt", path=path_database)

# Asset_id and runtime data validation
errors = check_failure_data(df)

# DataFrame preparation:
# >>> Check/Remove columns with unique values
plot_columns_nunique(df, title=f"Paper CO - Unique values per variable", savefig=SAVEFIG)
df = remove_cols_unique(df, verbose=False)

# >>> Feature Engineering with runtime as countdown to failure
df = feat_eng_runtime_inv(df)


# EDA Plots
# Due dataprocessing visualization ONLY, using a slice of 25% of data
df_sample = df.sample(frac=.25, random_state=314)

# >>> Univariate analysis
for col in df.columns:
    plot_histbox(data=df_sample[col], title=f"Paper CO - Histogram - {col}", savefig=SAVEFIG)

# >>> Bivariate analysis
cols_comb = list(itertools.combinations(list(df.columns), 2))
for (var_x, var_y) in cols_comb:
    plot_scatterhist(x=df_sample[var_x], y=df_sample[var_y], xlabel=var_x, ylabel=var_y, mark_size=10,
                     title=f"Paper CO - Scatter - {var_x} vs {var_y}", savefig=SAVEFIG)

plot_heatmap(df_sample, title="Paper CO - Heatmap", savefig=SAVEFIG)


# Scout theme: "Always leave the campsite cleaner than you found it"
organize_report(src=path_main, dst="report")

