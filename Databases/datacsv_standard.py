
# Libraries
import os
import numpy as np
import pandas as pd


# Functions
def files_csv():
    """
    Returns all .csv files in the current folder.

    """
    files_raw = os.listdir()
    files_csv = list()

    for f in files_raw:
        name, ext = f.split(".")

        if(ext == "csv"):
            files_csv.append(f)


    return files_csv


def rename_cols(DataFrame):
    """
    Apply a standard for columns names.

    """
    data = DataFrame.copy()
    updated = False

    # Remove unused columns
    if(data.columns.to_list().count("Unnamed: 0") == 1):
        data = data.drop(columns=["Unnamed: 0"])

    # columns with standard
    new_names = dict()
    for col in data.columns:
        new_col = col[:]

        if(new_col.isupper == True):
            new_col = new_col.lower()
        
        new_col = new_col.replace("-", "_")\
                         .replace(" ", "_")\
                         .replace("(", "")\
                         .replace(")", "")

        if(new_col != col):
            new_names[col]: new_col


    if(len(new_names) > 0):
        data = data.rename(columns=new_names)
        updated = True


    return data, updated


# Program
files = files_csv()

for f in files:
    data = pd.read_csv(f, sep=",", encoding="utf-8")
    data, updated = rename_cols(data)

    if(updated == True):
        data.to_csv(f, index=False, sep=",", encoding="utf-8")
        print(f" > file updated: '{f}'")


# end
