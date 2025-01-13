# P472 - GEFCom2014 Project (Time Series)

# Libraries
import os
import sys

import datetime as dt
import numpy as np
import pandas as pd


# Setup/Config



# Functions
def load_dataset():
    """
    Load the complete **dataset** from GEFCom2014 project.
    Focused in Time Series learning methods.

    """
    # Load data
    filename = "GEFCom2014-E.csv"
    data = pd.read_csv(filename, index_col=0, sep=",", encoding="utf-8")

    # Prepare columns names
    col_names = dict()
    for old_name in data.columns:
        new_name = old_name.lower()
        col_names[old_name] = new_name

    data = data.rename(columns=col_names)
    data = data.rename(columns={"t": "temperature"})

    # Prepare date and time
    data["hour"] = data["hour"].apply(_hour_to_string)  
    data["datetime"] = data["date"] + " " + data["hour"]
    data["datetime"] = pd.to_datetime(data["datetime"], format="%Y-%m-%d %H:%M")
    data.index = data["datetime"]
    data = data.drop(columns=["date", "hour", "datetime"])

    
    return data


def data_preparation(DataFrame, start_time, end_time):
    """
    Slices the DataFrame between **start_time** and **end_time**.
    Important: Inclusive in both sides: [start_time, end_time].

    """
    # Date and Time prep
    for var in [start_time, end_time]:
        var = pd.to_datetime(var, format="%Y-%m-%d")

    # DataFrame slicing [start_time, end_time]
    DataFrame = DataFrame[DataFrame.index >= start_time]
    DataFrame = DataFrame[DataFrame.index <= end_time]

    return DataFrame


def _hour_to_string(hour):
    """
    Internal function.
    Transforms an hour informed as an integer(1; 24) and returns the
    hour as a string pre-formated to be used as datetime stamp (HH:MM)
    and (00:00; 23:59)

    """
    # Substitute hour 24 per 0
    if(hour == 24):
        hour = 0

    # Insert a zero suffix for two digits padded hour
    if(hour <= 9):
        hour_suffix = "0"
    else:
        hour_suffix = ""

    # Create %HH:%MM string
    string = f"{hour_suffix}{hour}:00"

    return string


def check_nans(DataFrame, columns="all", decimals=2, verbose=True):
    """
    Check the number of NaNs (Not a Number) in given **columns**.
    Columns could be a list if desired some specific columns to check.

    """
    # Columns preparation
    if(columns == "all" or columns == None):
        columns = list(DataFrame.columns)


    # NaNs check
    nans_register = dict()

    no_rows = DataFrame.shape[0]
    for col in columns:
        no_nans = DataFrame[col].isna().sum()
        pct_nans = np.round((no_nans / no_rows) * 100, decimals=decimals)

        nans_register[col] = [no_nans, pct_nans]
        print(f" > {col}: NaNs= {no_nans} ({pct_nans}%)")

    # [1] Create an function to print it more friendly
    # [2] Create a plot to receive this information and transform it
    # into a plot
    
    return nans_register
         

# Program --------------------------------------------------------------
df = load_dataset()
df = data_preparation(df, "2012-01-01", "2014-12-31")

    
# end
