# Project Paper CO

# Libraries
import os
import itertools
import warnings

from fractions import Fraction

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt


# Project libraries
from src.paper_co_tools import (load_dataset, remove_cols_unique, feat_eng_runtime_inv,
                                add_failure_tag, remove_df_columns, organize_report)

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

    cols_remove = ["asset_id", "runtime", "tag_6", "runtime_inv","setting_1", "setting_2"]
    data = remove_df_columns(data, columns=cols_remove)

    return data


def holdout_split(DataFrame, target, test_size=0.2, random_state=42):
    """
    Splits DataFrame into **two** Dataframes for Train, Validation and Test.

    Arguments:
    * DataFrame:
    * target: Target for Stratified KFold keep the proportions constant,
    * test_size: Size of test dataframe,
    * random_state: Seed for tests reproducibility,

    Returns:
    * trainval: Pandas DataFrame to be used for Train and Validation
    * test: Pandas DataFrame to be used for Test.

    More info about the holdout strategy:
    

    """
    # test_size preparation
    test_size = np.round(test_size, decimals=2)
    test_size = Fraction(test_size).limit_denominator()

    splits_get = test_size.numerator
    splits_total = test_size.denominator  


    # Stratified KFold
    skf = StratifiedKFold()

    # Hyperparams
    skf.n_splits = splits_total
    skf.shuffle = True
    skf.random_state = random_state

    # Target split
    x = DataFrame.drop(columns=[target])
    y = DataFrame[target]

    # trainval and test split
    trainval = np.array(DataFrame.index)
    test = np.array([])
    
    for i, [_, test_index] in enumerate(skf.split(x, y)):
        test = np.append(test, test_index)

        # test array is an array to be removed from trainval array.
        items_to_keep = ~np.isin(trainval, test)
        trainval = trainval[items_to_keep]
        

        if(i >= (splits_get - 1)):
            break

    # DataFrame split
    data_trainval = DataFrame.loc[trainval, :]
    data_test = DataFrame.loc[test, :]


    return data_trainval, data_test
        


    

    



   
# Setup/Config ----------------------------------------------------------
path_main = os.getcwd()
path_database = os.path.join(path_main, "database")
path_report = os.path.join(path_main, "report")

warnings.filterwarnings("ignore")

SAVEFIG = True


# Program ---------------------------------------------------------------

# Import dataset for Models
df = prepare_dataset(filename="pm_train.txt", path=path_database)
target = "failure_flag"

df_trainval, df_test = holdout_split(df, target, test_size=.2, random_state=314)




# Scout theme: "Always leave the campsite cleaner than you found it"
organize_report(src=path_main, dst="report")

