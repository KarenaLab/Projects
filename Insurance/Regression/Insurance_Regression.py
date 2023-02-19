
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

from color_source import *
from plot_histogram import *
from plot_heatmap import *
from one_hot_encoding import *


# Setup/Config
path_main = os.getcwd()
path_report = os.path.join(path_main, "Report")

pd.set_option('display.precision', 3)


# Functions
filename = "insurance.csv"
df = pd.read_csv(filename, sep=",", encoding="utf-8")

cols_numeric = ["age", "bmi", "charges"]
cols_categoric = ["sex", "children", "smoker", "region"]


# New features (test)
charges_by_head_list = []
for i in range(0, df.shape[0]):
    charges_by_head = df.loc[i, "charges"] / (df.loc[i, "children"] + 1)
    charges_by_head_list.append(charges_by_head)

df["charges_by_head"] = charges_by_head_list


#df = one_hot_encoding(df, columns=cols_categoric)


#for col in cols_numeric:
#    os.chdir(path_report)
#    plot_histogram(df[col], title=f"Insurance - {col}", savefig=False) 
    
"""
x_axis = "age"
y_axis = "charges"
c_axis = "smoker"

palette = list(romania_palette().values())
c_values = df[c_axis].unique().tolist()
color_dict = dict(zip(c_values, palette[0:len(c_values)]))

color = df[c_axis].map(color_dict)

title = f"Insurance - {x_axis} versus {y_axis} by {c_axis}"

# Plot
fig = plt.figure(figsize=[8, 4.5])
plt.suptitle(title, fontsize=10, fontweight="bold")

plt.scatter(df[x_axis], df[y_axis], c=color, s=30, edgecolor="white", alpha=0.6, zorder=20)

plt.ylabel(y_axis, loc="top")
plt.xlabel(x_axis, loc="right")
plt.grid(axis="both", color="lightgrey", linestyle="--", linewidth=0.5, zorder=10)

plt.show()
"""




