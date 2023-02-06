
# Libraries
import os
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Config/Setup
path_script = os.getcwd()
path_report = r"D:\01 - Projects Binder\161 - Project - Credit_Card_Default\Report"


# Functions
def col_preparation(DataFrame, method="lower", verbose=True):
    """
    Standartize columns names.

    Variables:
    * DataFrame = DataFrame to be standartized,
    * method = lower, UPPER or Title. (default=lower),
    * verbose = True or False. (default=True).

    """
    data = DataFrame.copy()
    method = method.lower()

    for col in data.columns:
        new_col = col[:]

        items_to_replace = {"-": "_",
                            " ": "_",
                            "(": "",
                            ")": ""}
        # add new items here: item to be replaced and new item.

        for old, new in list(zip(items_to_replace.keys(), items_to_replace.values())):
            new_col = new_col.replace(old, new)

            if(method == "lower"):
                new_col = new_col.lower()

            elif(method == "upper"):
                new_col = new_col.upper()

            elif(method == "title"):
                new_col = new_col.title()        


        data = data.rename(columns={col: new_col})

        if(verbose == True):
            print(f" > column {col} renamed for **{new_col}**")


    if(verbose == True):
        print("")
        
    return data

def remove_duplicates(DataFrame, verbose=True):
    """


    """
    data = DataFrame.copy()

    no_rows = data.shape[0]
    duplicated = np.array(data.index[df.duplicated()])
    data = data.drop_duplicates(ignore_index=True)

    if(verbose == True):
        print(f" > Duplicated items removed: {len(duplicated)} ({(len(duplicated)/no_rows)*100:.3f})%\n")

    return data
    

# Program --------------------------------------------------------------

# Load dataset
filename = "creditcard_clients_2005.csv"
df = pd.read_csv(filename, sep=",", encoding="utf-8")

# Dataset preparation
df = col_preparation(df)
df = remove_duplicates(df)
