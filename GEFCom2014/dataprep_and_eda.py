# P472 - GEFCom2014 Project (Time Series)

# Libraries
import os
import sys

import numpy as np
import pandas as pd


# Setup/Config



# Functions
def load_dataset():
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

    # Prepare data
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")


    return data


# Program --------------------------------------------------------------
df = load_dataset()

    

    
    
