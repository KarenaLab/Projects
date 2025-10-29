# [P504] Cement compressive strength
# Performs EDA (Exploratory Data Analysis)

# Libraries
import os
import shutil
import itertools

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt


# Personal modules
from concrete_tools import load_dataset
from src.plot_histbox import plot_histbox
from src.plot_barh import plot_barh
from src.plot_scatterhist import plot_scatterhist


# Functions
def cols_numerical():
    cols = ["cement_kg_p_m3", "blast_furnace_slag_kg_p_m3",
            "fly_ash_kg_p_m3", "water_kg_p_m3", "superplasticizer_kg_p_m3",
            "coarse_aggregate_kg_p_m3", "fine_aggregate_kg_p_m3"]

    return cols


def cols_categorical():
    cols = ["age_days"]

    return cols


def cols_variable():
    cols = cols_numerical() + cols_categorical()

    return cols
    

# Setup/Config


def cross_target(variables, target):
    combination = list()
    for i in variables:
        info = (i, target)
        combination.append(info)


    return combination


def organize_report(path=None):
    # Path
    path_back = os.getcwd()
    if(path != None):
        os.chdir(path)

    # Move
    for f in os.listdir():
        name, extension = os.path.splitext(f)

        if(extension == ".png"):
            src = os.path.join(os.getcwd(), f)
            dst = os.path.join(os.getcwd(), "report", f)
            shutil.move(src, dst)

    os.chdir(path_back)

    return None


def series_to_count(Series):
    counter = dict()

    for i in Series:
        if i in counter:
            counter[i] = counter[i] + 1

        else:
            counter[i] = 1

       
    return counter


# Setup/Config
savefig = False



# Program --------------------------------------------------------------
df = load_dataset()
target = "compressive_strength_mpa"

# Univariate analysis
for col in cols_numerical():
    #plot_histbox(df[col], title=f"Cement Strength - HistBox - {col}", savefig=savefig)
    pass   
    
for col in cols_categorical():
    info = series_to_count(df[col])
        

# Bivariate analysis
var_comb = list(itertools.combinations(cols_numerical(), 2))
for var_x, var_y in var_comb:
    plot_scatterhist(x=df[var_x], y=df[var_y], title=f"Cement Strengh - {var_x} vs {var_y}",
                     xlabel=var_x, ylabel=var_y, mark_size=15, savefig=True)
    

# Variables versus target
var_comb = cross_target(cols_variable(), target)
for var_x, var_y in var_comb:
    plot_scatterhist(x=df[var_x], y=df[var_y], title=f"Cement Strengh - {var_x} vs {var_y}",
                     color="darkred", xlabel=var_x, ylabel=var_y, mark_size=15, savefig=True)
    

# Heatmap


# Insights


# Organize folder
organize_report()

# end
