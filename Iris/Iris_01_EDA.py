
# Libraries
import os

import numpy as np
import pandas as pd


# Personal Modules
import sys
sys.path.append(r"C:\python_modules")

from data_preparation import *
from plot_histogram import *
from plot_heatmap import *
from plot_histogramduo import *


# Setup/Config
savefig = False


# Program --------------------------------------------------------------
filename = "iris.csv"
df = pd.read_csv(filename, sep=",", encoding="utf-8")

df = col_preparation(df, method="lower")
df = remove_duplicates(df)
df = nan_counter(df)
_ = distinct_counter(df)


# EDA
cols_numeric = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

for col in cols_numeric:
    plot_histogram(df[col], title=f"Iris - Histogram - {col}", savefig=savefig)

for col in cols_numeric:
    data_setosa = df.groupby(by="species").get_group("setosa")[col].reset_index(drop=True)
    data_versicolor = df.groupby(by="species").get_group("versicolor")[col].reset_index(drop=True)  
    data_virginica = df.groupby(by="species").get_group("virginica")[col].reset_index(drop=True)

    plot_histogramduo(serie1=data_setosa, name1="setosa",
                      serie2=data_versicolor, name2="versicolor",
                      serie3=data_virginica, name3="virginica",
                      title=f"Iris - Histogram multiple - {col}", savefig=savefig)

plot_heatmap(df, columns=cols_numeric, title=f"Iris - Heatmap", savefig=savefig)


