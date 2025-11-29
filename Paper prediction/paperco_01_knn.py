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
from src.paper_co_tools import (load_dataset, remove_cols_unique, feat_eng_runtime_inv,
                                add_failure_tag, organize_report)

from src.plot_histbox import plot_histbox
from src.plot_barv import plot_barv
from src.plot_scatterhist import plot_scatterhist
from src.plot_heatmap import plot_heatmap


# Functions
def prepare_dataset(filename, path):
    """
    Import and prepare **DataFrame** for modeling.

    Arguments:
    * filename: string with filename and extension,
    * path: (Optional) If file is not in root path,

    Return:
    * data: Pandas Dataframe
    
    """
    data = load_dataset(filename=filename, path=path)
    data = remove_cols_unique(data, verbose=False)
    data = feat_eng_runtime_inv(data)
    data = add_failure_tag(data, threshold=-20)

    cols_remove = ["asset_id", "runtime", "tag_6", "runtime_inv"]
    data = remove_columns(data, columns=cols_remove)

    return data


def remove_columns(DataFrame, columns, verbose=False):
    """
    Remove columns that will not be helpful for the models.

    Arguments:
    * DataFrame: Pandas dataframe of project,
    * columns: Columns names to be removed from,
    * verbose: True or False* (default=False),

    Return:
    * DataFrame: Processed Pandas DataFrame

    """
    # Columns names preparation
    cols_remove = list()
    cols_dataframe = list(DataFrame.columns)

    for col in columns:
        if(cols_dataframe.count(col) == 1):
            cols_remove.append(col)

        else:
            if(verbose == True):
                print(f" > column '{col}' does NOT exists")

    if(len(cols_remove) > 0):
        DataFrame = DataFrame.drop(columns=cols_remove)


    return DataFrame
        
    

   
# Setup/Config ----------------------------------------------------------
path_main = os.getcwd()
path_database = os.path.join(path_main, "database")
path_report = os.path.join(path_main, "report")

SAVEFIG = True

warnings.filterwarnings("ignore")


# Program ---------------------------------------------------------------

# Import dataset for Models
df = prepare_dataset(filename="pm_train.txt", path=path_database)






# Scout theme: "Always leave the campsite cleaner than you found it"
organize_report(src=path_main, dst="report")

