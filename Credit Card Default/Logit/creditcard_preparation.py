
# Libraries
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Personal modules
sys.path.append(r"C:\python_modules")
from data_preparation import *
from plot_histogram import *
from plot_missingdata_col import *
from plot_missingdata_row import *
from plot_heatmap import *
from plot_scatterhist import *


# Setup/Config
seed = 27
np.random.seed(seed)

path_main = os.getcwd()
path_database = os.path.join(path_main, "Database")
path_report = os.path.join(path_main, "Report")


# Functions
def read_csv(path, filename, sep=",", encoding="utf-8"):
    """
    Internal function for reading database from folder/filename.

    """
    path_origin = os.getcwd()

    os.chdir(path)   
    data = pd.read_csv(filename, sep=sep, encoding=encoding)
    os.chdir(path_origin)

    return data


def cols_logit():
    """
    Shortcut for columns to keep.

    """
    columns = ["limit_bal", "education", "marriage", "age", "pay_1",
               "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4",
               "bill_amt5", "bill_amt6",
               "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4",
               "pay_amt5", "pay_amt6",
               "default_pmt"]

    return columns


# Program
print(" ****  Credit Card 2005  **** \n")

df = read_csv(path_database, filename="creditcard_clients_2005.csv")
df = col_preparation(df)
df = remove_duplicates(df)
df = df[cols_logit()]

plot_heatmap(df, title="Credit Card 2005 Heatmap")

for col in cols_logit():
    plot_histogram(df[col], title=f"Credit Card 2005 - {col}", bins="rice") 



