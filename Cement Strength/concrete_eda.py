# [P504] Cement compressive strength
# Performs EDA (Exploratory Data Analysis)

# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt


# Personal modules
from concrete_tools import load_dataset


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



# Program --------------------------------------------------------------
df = load_dataset()
target = "compressive_strength_mpa"

# Univariate analysis
for col in cols_numerical():
    pass

for col in cols_categorical():
    pass

# Bivariate analysis


# Variables versus target


# Heatmap


# Insights


# end
