# [P504] Cement compressive strength
# Linear Regression model with seed variation

# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (mean_absolute_error, root_mean_squared_error,
                             r2_score)

import matplotlib.pyplot as plt


# Personal modules
from concrete_tools import (load_dataset, cols_variable, prep_pipeline,
                            aggregate_results, organize_report)

from src.plot_histbox import plot_histbox


# Functions
def create_kfold(DataFrame, n_splits=5, random_state=None):
    # Shuffle and random state
    if(isinstance(random_state, int) == True):
        shuffle = False

    else:
        shuffle = True
        
    # Split folds
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    folds = dict()
    for i, (train_index, test_index) in enumerate(kf.split(DataFrame)):
        folds[i] = [train_index, test_index]


    return folds


def split_fold(DataFrame, target, train_index, test_index):
    # Target split
    variables = list(DataFrame.columns)
    variables.remove(target)
    
    # Train and Test split
    x_train = DataFrame.loc[train_index, variables]
    x_test = DataFrame.loc[test_index, variables]

    y_train = DataFrame.loc[train_index, target]
    y_test = DataFrame.loc[test_index, target]

    
    return x_train, x_test, y_train, y_test
    

def scaler(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)

    cols = x_train.columns
    for i in [x_train, x_test]:
        i = scaler.transform(i)
        i = pd.DataFrame(data=i, columns=cols)

    return x_train, x_test


def model_linregr(x_train, x_test, y_train, y_test=None):
    regr = LinearRegression()

    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    params = dict()
    params["coef"] = regr.coef_
    params["intercept"] = regr.intercept_

    return y_pred, params


def regr_metrics(y_true, y_pred):
    results = dict()
    results["mae"] = mean_absolute_error(y_true, y_pred)
    results["rmse"] = root_mean_squared_error(y_true, y_pred)
    results["r2_score"] = r2_score(y_true, y_pred)
    results["pearson_r"] = stats.pearsonr(y_true, y_pred).statistic

    return results


def summarize_results(array):
    summary = dict()

    # Cluster same metric
    for i in array:
        for key, value in i.items():
            if key not in summary:
                summary[key] = [value]

            else:
                summary[key].append(value)

    # Calculate the mean (average) of each metric
    for key, value in summary.items():
        summary[key] = np.mean(summary[key])


    return summary


def pipeline(DataFrame, target, random_state=None):
    # Data split
    folds = create_kfold(df, random_state=random_state)
    results = list()
    
    for i in folds.keys():
        train_index = folds[i][0]
        test_index = folds[i][1]
        
        x_train, x_test, y_train, y_test = split_fold(df, target, train_index, test_index)
        x_train, x_test = scaler(x_train, x_test)

        # Model: Linear Regression
        y_pred, _ = model_linregr(x_train, x_test, y_train)
        fold_results = regr_metrics(y_test, y_pred)
        results.append(fold_results)


    results = summarize_results(results) 

    return results


def check_results(results):
    for col in results.columns:
        print(f" *** {col} ***")
        print(f"   Mean: {results[col].mean()}")
        print(f" StdDev: {results[col].std()}")
        print("")

    return None


def add_feateng(DataFrame, remove_old=False):
    # Fine and Coarse Ratio
    DataFrame["fc_ratio"] = DataFrame["fine_aggregate_kg_p_m3"] / DataFrame["coarse_aggregate_kg_p_m3"]

    # Aggregate and Cement Ratio
    DataFrame["aggcmt_ratio"] = (DataFrame["fine_aggregate_kg_p_m3"] + DataFrame["coarse_aggregate_kg_p_m3"]) / DataFrame["cement_kg_p_m3"]

    # Water and Cement Ratio
    DataFrame["wtrcmt_ratio"] = DataFrame["water_kg_p_m3"] / DataFrame["cement_kg_p_m3"]

    if(remove_old == True):
        cols_remove = ["fine_aggregate_kg_p_m3", "coarse_aggregate_kg_p_m3", "water_kg_p_m3", "cement_kg_p_m3"]
        DataFrame = DataFrame.drop(columns=cols_remove)
            

    return DataFrame

                 
# Setup/Config
savefig = False

    
# Program --------------------------------------------------------------
df = load_dataset()
df = add_feateng(df, remove_old=False)

target = "compressive_strength_mpa"
seed = 1
size = 100

# Modeling
np.random.seed(seed)

df_results = pd.DataFrame(data=[])
for seed in np.random.randint(low=0, high=500, size=size):
    results = pipeline(df, target, random_state=seed)

    for key, value in results.items():
        df_results.loc[seed, key] = value

# Print
for col in df_results.columns:  
    plot_histbox(df_results[col], title=f"Concrete Strength - Model - LinRegr - CV 5 folds - {col}",
                 savefig=savefig)


check_results(df_results)    

   
# Scout theme: "Always leave the campsite cleaner than you found it"
organize_report()
