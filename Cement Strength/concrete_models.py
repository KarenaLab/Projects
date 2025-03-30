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
from src.split_train_test import split_train_test

# Functions
def cols_variable():
    cols = ["cement_kg_p_m3", "blast_furnace_slag_kg_p_m3",
            "fly_ash_kg_p_m3", "water_kg_p_m3", "superplasticizer_kg_p_m3",
            "coarse_aggregate_kg_p_m3", "fine_aggregate_kg_p_m3", "age_days"]

    return cols


# Setup/Config




# Program --------------------------------------------------------------
df = load_dataset()
target = "compressive_strength_mpa"

# Train/Validation/Test strategy
trainval, test = split_train_test(df, train_size=70, seed=314)





# end
