# Project Paper CO

# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt



# Functions ------------------------------------------------------------
def load_txt(filename, path=None, sep=" "):
    """
    Import project dataset with .txt format.
    
    """
    path_back = os.getcwd()
    if(path != None):
        filename = os.path.join(path, filename)

    data = pd.read_csv(filename, header=None, sep=sep)


    return data


def remove_nan_cols(DataFrame):
    
    cols_remove = list()
    for col in DataFrame.columns:
        not_nan = DataFrame[col].notna().sum()

        if(not_nan == 0):
            cols_remove.append(col)

    DataFrame = DataFrame.drop(columns=cols_remove)


    return DataFrame
        

def prepare_columns(DataFrame):
    cols = ["asset_id", "runtime"]

    # Setting columns
    for i in range(1, 3+1):
        tag = f"setting_{i}"
        cols.append(tag)

    # Tag columns
    for i in range(1, 21+1):
        tag = f"tag_{i}"
        cols.append(tag)

    # Rename DataFrame
    cols_rename = dict()
    for old_name, new_name in zip(list(DataFrame.columns), cols):
        cols_rename[old_name] = new_name

    DataFrame = DataFrame.rename(columns=cols_rename)
    

    return DataFrame


def check_duplicates(DataFrame):
    size_before = DataFrame.shape[0]

    DataFrame = DataFrame.drop_duplicates()
    size_after = DataFrame.shape[0]

    if(size_before > size_after):
        print(f" > Rows duplicated removed: {size_before - size_after}")


    return DataFrame


def load_dataset(filename, path=None):
    data = load_txt(filename=filename, path=path)
    data = remove_nan_cols(data)
    data = prepare_columns(data)
    data = check_duplicates(data)


    return data


# end
