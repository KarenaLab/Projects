# [P504] Cement compressive strength
# Performs EDA (Exploratory Data Analysis)

# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (mean_absolute_error, root_mean_squared_error,
                             r2_score)

import matplotlib.pyplot as plt


# Personal modules
from concrete_tools import (load_dataset, cols_variable, prep_pipeline,
                            aggregate_results, organize_report)

from src.plot_linemultiple import plot_linemultiple


# Functions
def split_target(DataFrame, target):
    x = DataFrame.drop(columns=[target])
    y = DataFrame[target]

    return x, y


def scaler(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)

    cols = x_train.columns
    for i in [x_train, x_test]:
        i = scaler.transform(i)
        i = pd.DataFrame(data=i, columns=cols)

    return x_train, x_test


def regr_kneigbors(x_train, x_test, y_train, n_neighbors=2):
    regr = KNeighborsRegressor()
    # Main parameters: n_neighbors, weights, p,

    # Hyperparams
    regr.n_neighbors = n_neighbors

    # Fit and predict
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    # Parameters
    params = dict()


    return y_pred, params


def neighbors_range(max_range=25, reverse=False):
    # Range (with only odd values)
    values = [i for i in range(2, max_range+1) if i%2 == 1]

    return values
    

def regr_metrics(y_true, y_pred):
    results = dict()
    results["mae"] = mean_absolute_error(y_true, y_pred)
    results["rmse"] = root_mean_squared_error(y_true, y_pred)
    results["r2_score"] = r2_score(y_true, y_pred)
    results["pearson_r"] = stats.pearsonr(y_true, y_pred).statistic

    return results


def find_best_hyperp(DataFrame, metric, best):
    """


    """
    # Data preparation
    DataFrame = DataFrame.dropna()

    # Set best parameters   
    if(best == "lower"):
        best_metric = np.inf

    elif(best == "upper"):
        best_metric = 0

    # Finder
    best_param = None

    for i in DataFrame.index:
        value = DataFrame.loc[i, metric]
        if(best == "lower" and value < best_metric):
            best_metric = value
            best_param = i

        elif(best == "upper" and value > best_metric):
            best_metric = value
            best_param = i

        else:
            break


    return best_param


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
df = add_feateng(df, remove_old=True)
target = "compressive_strength_mpa"

# Data Split
x, y = split_target(df, target=target)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,
                                                    random_state=27)

# Scaler
x_train, x_test = scaler(x_train, x_test)

# Model: Ridge (L2)

df_results = pd.DataFrame(data=[])
for n in range(2, 7+1):
    y_pred, _ = regr_kneigbors(x_train, x_test, y_train, n_neighbors=n)
    results = regr_metrics(y_test, y_pred)    

    for key, value in results.items():
        df_results.loc[n, key] = value


df_results["hmean"] = df_results.apply(lambda row: stats.hmean([row["mae"], row["rmse"]]), axis=1)


best_n = find_best_hyperp(df_results, metric="rmse", best="lower")
y_pred, _ = regr_kneigbors(x_train, x_test, y_train, n_neighbors=best_n)
results = regr_metrics(y_test, y_pred)

# Plots
plot_linemultiple(df_results, columns=["mae", "rmse", "hmean_error"], xlabel="n_neighbors",
                  title=f"Concrete Strength - Model - KNN Regressor - Detail", savefig=savefig)

                  
# Scout theme: "Always leave the campsite cleaner than you found it"
organize_report()

