# Project Paper CO

# Libraries
import os
import itertools

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


    """
    cols_remove = list()
    for col in DataFrame.columns:
        if(DataFrame[col].nunique() == 1):
            cols_remove.append(col)

            if(verbose == True):
                print(f" > Column '{col}' removed due unique item")


    DataFrame = DataFrame.drop(columns=cols_remove)

    return DataFrame


def feat_eng_runtime_inv(DataFrame):
    
    for asset in DataFrame["asset_id"].unique():
        info = DataFrame.groupby(by="asset_id").get_group(asset)
        runtime_max = info["runtime"].max()
        info["runtime_inv"] = info["runtime"].apply(lambda x: x - runtime_max)

        for i in info.index:
            DataFrame.loc[i, "runtime_inv"] = info.loc[i, "runtime_inv"]


    return DataFrame
    

        
  
# Setup/Config
path_main = os.getcwd()
path_database = os.path.join(path_main, "database")
path_report = os.path.join(path_main, "report")

SAVEFIG = True


# Program ---------------------------------------------------------------
df = load_dataset(filename="pm_train.txt", path=path_database)

# Asset_id and runtime data validation
errors = check_failure_data(df)


# DataFrame preparation
cols_unique = pd.DataFrame(data=[])
for col in df.columns:
    cols_unique.loc[col, "unique"] = df[col].nunique()

#plot_barv(x=cols_unique.index, height=cols_unique["unique"], xrotation=True,
#          title=f"Paper CO - Unique values per variable", savefig=SAVEFIG)

df = remove_cols_unique(df, verbose=False)
df = feat_eng_runtime_inv(df)
df_sample = df.sample(frac=.25, random_state=314)


"""
# Univariate analysis
for col in df.columns:
    plot_histbox(data=df_sample[col], title=f"Paper CO - Histogram - {col}", savefig=SAVEFIG)


# Bivariate analysis
cols_comb = list(itertools.combinations(list(df.columns), 2))
for (var_x, var_y) in cols_comb:
    plot_scatterhist(x=df_sample[var_x], y=df_sample[var_y], xlabel=var_x, ylabel=var_y, mark_size=15,
                     title=f"Paper CO - Scatter - {var_x} vs {var_y}", savefig=SAVEFIG)

plot_heatmap(df_sample, title="Paper CO - Heatmap", savefig=SAVEFIG)
"""


# Scout theme: "Always leave the campsite cleaner than you found it"
organize_report(src=path_main, dst="report")

