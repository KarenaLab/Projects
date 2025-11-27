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
    # Create columns names
    cols = list()
    cols = cols + cols_control()         # Control
    cols = cols + cols_settings()        # Settings
    cols = cols + cols_tags()            # Tag/Sensor

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


def cols_control():
    cols = ["asset_id", "runtime"]

    return cols


def cols_settings():
    cols = list()   
    for i in range(1, 3+1):
        tag = f"setting_{i}"
        cols.append(tag)

    return cols


def cols_tags():
    cols = list()
    for i in range(1, 21+1):
        tag = f"tag_{i}"
        cols.append(tag)

    return cols


def organize_report(src=None, dst="", verbose=False):
    # Path
    path_back = os.getcwd()
    if(src != None):
        os.chdir(path)

    # Move
    for f in os.listdir():
        name, extension = os.path.splitext(f)

        if(extension == ".png"):
            path_src = os.path.join(os.getcwd(), f)
            path_dst = os.path.join(os.getcwd(), dst, f)
            shutil.move(path_src, path_dst)

            if(verbose == True):
                print(f" > File '{f}' transfered for `\report`")
                

    os.chdir(path_back)

    return None

    
# end
