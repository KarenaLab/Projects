# Project Paper CO

# Libraries
import os
import shutil

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt



# Functions ------------------------------------------------------------
def load_txt(filename, path=None, sep=" "):
    """
    Import project dataset with .txt format.

    Arguments:
    * filename: string with filename and extension,
    * path: (Optional) If file is not in root path,
    * sep: Separator for columns (default=" ", space),

    Return:
    * data: Pandas Dataframe
    
    """
    # Path preparation
    path_back = os.getcwd()
    if(path != None):
        filename = os.path.join(path, filename)

    # Data import
    data = pd.read_csv(filename, header=None, sep=sep)


    return data


def remove_nan_cols(DataFrame):
    """
    Corretion for DataFrame with aditional spaces in the end,
    will delete/remove column if it is 100% of NaNs.

    Arguments:
    * DataFrame: Pandas DataFrame

    Return:
    * DataFrame: Processed Pandas DataFrame

    """
    
    cols_remove = list()
    for col in DataFrame.columns:
        not_nan = DataFrame[col].notna().sum()

        if(not_nan == 0):
            cols_remove.append(col)


    if(len(cols_remove) > 0):
        DataFrame = DataFrame.drop(columns=cols_remove)


    return DataFrame
        

def prepare_columns(DataFrame):
    """
    Add columns names for DataFrame according to exercise
    description.

    Important: Using calling functions for columns, do not edit it
    here.

    """
    # Create columns names
    cols = list()
    cols = cols + cols_control()         # Control
    cols = cols + cols_settings()        # Settings
    cols = cols + cols_tags()            # Tag/Sensor

    # Columns names correspondency
    cols_rename = dict()
    for old_name, new_name in zip(list(DataFrame.columns), cols):
        cols_rename[old_name] = new_name


    DataFrame = DataFrame.rename(columns=cols_rename)
    
    return DataFrame


def check_duplicates(DataFrame):
    """
    Check if **DataFrame** has duplicated rows.
    Function has a locked verbose mode.
    
    Arguments:
    * DataFrame: Pandas DataFrame

    Return:
    * DataFrame: Processed Pandas DataFrame

    """
    # Initial DataFrame size
    size_before = DataFrame.shape[0]
    DataFrame = DataFrame.drop_duplicates()

    # Verification and soft warning for user
    size_after = DataFrame.shape[0]
    if(size_before > size_after):
        print(f" > Rows duplicated removed: {size_before - size_after}")


    return DataFrame


# Export functions ------------------------------------------------------
def load_dataset(filename, path=None, sep=" "):
    """
    Import dataset with corrections.

    Arguments:
    * filename: string with filename and extension,
    * path: (Optional) If file is not in root path,
    * sep: Separator for columns (default=" ", space),

    Return:
    * data: Pandas dataset without data processing (for EDA)

    """
    data = load_txt(filename=filename, path=path, sep=sep)
    data = remove_nan_cols(data)
    data = prepare_columns(data)
    data = check_duplicates(data)

    return data


def cols_control():
    """
    First **two** columns of DataFrame described in exercise document
    (table 01, page 02).

    Return:
    * cols: List with columns names string.
    
    """
    cols = ["asset_id", "runtime"]

    return cols


def cols_settings():
    """
    **Three** columns settings of DataFrame described in exercise document
    (table 01, page 02).

    Return:
    * cols: List with columns names string.
    
    """
    cols = list()   
    for i in range(1, 3+1):
        tag = f"setting_{i}"
        cols.append(tag)


    return cols


def cols_tags():
    """
    **Twenty one** columns tags/sensors of DataFrame described in exercise
    document (table 01, page 02).

    Return:
    * cols: List with columns names string.
    
    """
    cols = list()
    for i in range(1, 21+1):
        tag = f"tag_{i}"
        cols.append(tag)


    return cols


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


def organize_report(src=None, dst="", verbose=False):
    """
    Move plots figures saved as .png saved in path* and move
    for a **destiny (dst)** path.

    Arguments:
    * src: (Optional) Path of the source to find the figures .png files,
    * dst: (Mandatory) Path destiny for the move.
    * verbose: True or False* (default=False)

    """
    # Path
    path_back = os.getcwd()
    if(src != None):
        os.chdir(src)

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
