# [P504] Cement compressive strength
# Performs EDA (Exploratory Data Analysis)

# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (r2_score, mean_absolute_error)


import matplotlib.pyplot as plt


# Personal modules
from concrete_tools import (load_dataset, cols_variable, prep_pipeline,
                            aggregate_results)

from src.split_train_test import split_train_test
from src.stratified_continuous_kfold import stratified_continuous_kfold

from src.plot_histbox import plot_histbox


# Functions

        
# Setup/Config

    
# Program --------------------------------------------------------------
df = load_dataset()
target = "compressive_strength_mpa"

# Train/Validation/Test strategy
trainval, test = split_train_test(df, train_size=70, seed=314)
skf_folds = stratified_continuous_kfold(trainval, target=target) 
#           (fold_no, train_index, validation_index)


folds_results = pd.DataFrame(data=[])

for (i, train_index, test_index) in skf_folds:
    _, _, results = prep_pipeline(df, train_index, test_index, target)

    results["fold"] = i
    fold_results = aggregate_results(folds_results, results)


# end
