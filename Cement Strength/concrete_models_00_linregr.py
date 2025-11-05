# [P504] Cement compressive strength
# Performs EDA (Exploratory Data Analysis)

# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt


# Personal modules
from concrete_tools import (load_dataset, cols_variable, prep_pipeline,
                            aggregate_results)


# Functions
def split_target(DataFrame, target):
    x = DataFrame.drop(columns=[target])
    y = DataFrame[target]

    return x, y
        
        
# Setup/Config


    
# Program --------------------------------------------------------------
df = load_dataset()
target = "compressive_strength_mpa"

x, y = split_target(df, target=target)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=53)
