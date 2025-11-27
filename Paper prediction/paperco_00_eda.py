# Project Paper CO

# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt


# Project libraries
from src.paper_co_tools import (load_dataset, organize_report)
from src.plot_histbox import plot_histbox



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

    
# Setup/Config
path_main = os.getcwd()
path_database = os.path.join(path_main, "database")
path_report = os.path.join(path_main, "report")

SAVEFIG = False


# Program ---------------------------------------------------------------
df = load_dataset(filename="pm_train.txt", path=path_database)

# asset_id and runtime data validation
errors = check_failure_data(df)


# Single variable analysis
for col in df.columns:
    plot_histbox(data=df[col], title=f"Paper CO - Histogram - {col}", savefig=True)
    

# Scout theme: "Always leave the campsite cleaner than you found it"
organize_report(src=path_main, dst="report")

