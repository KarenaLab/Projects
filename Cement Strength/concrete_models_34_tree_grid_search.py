# [P504] Cement compressive strength
# Performs EDA (Exploratory Data Analysis)

# Libraries
import os
import itertools

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

from sklearn.metrics import (mean_absolute_error, root_mean_squared_error,
                             r2_score)

import matplotlib.pyplot as plt


# Personal modules
from concrete_tools import (load_dataset, cols_variable, prep_pipeline,
                            aggregate_results, organize_report)

from src.plot_lineduo import plot_lineduo
#from src.plot_contour import plot_contour


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


def regr_decisiontree(x_train, x_test, y_train,
                      criterion="squared_error",
                      max_depth=None, max_leaf_nodes=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0,
                      random_state=None,
                      title=None, showfig=True, savefig=False):

    # Model: Decison Tree
    # More info:
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    regr = DecisionTreeRegressor()

    # Hyperparams
    regr.criterion = criterion
    regr.max_depth = max_depth
    regr.max_leaf_nodes = max_leaf_nodes
    regr.min_samples_leaf = min_samples_leaf
    regr.min_samples_split = min_samples_split
    regr.min_weight_fraction_leaf = min_weight_fraction_leaf
    regr.random_state = random_state

    # Fit and predict
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    # Parameters
    params = dict()
    params["hyperparams"] = regr.get_params()
    params["feature_names_in"] = list(regr.feature_names_in_)
    params["feature_importances"] = list(regr.feature_importances_)
    params["tree_depth"] = regr.get_depth()
    params["tree_nodes"] = regr.tree_.node_count
    params["tree_leaves"] = regr.get_n_leaves()


    # Plot
    plot_tree(regr, proportion=True)

    if(title == None):
        title = "Decision Tree Regression"

    if(savefig == True):
        plt.savefig(title, dpi=320)
        print(f' > Plot saved as "{title}.png"')
        
    if(showfig == True):
        plt.show()

    plt.close()


    return y_pred, params
    

def regr_metrics(y_true, y_pred):
    results = dict()
    results["mae"] = mean_absolute_error(y_true, y_pred)
    results["rmse"] = root_mean_squared_error(y_true, y_pred)
    results["r2_score"] = r2_score(y_true, y_pred)
    results["pearson_r"] = stats.pearsonr(y_true, y_pred).statistic
    results["f_score"] = stats.hmean([results["mae"], results["rmse"]])
    
    return results



                   
# Setup/Config
path_main = os.getcwd()
savefig = False

    
# Program --------------------------------------------------------------
df = load_dataset()
target = "compressive_strength_mpa"

# Data Split
x, y = split_target(df, target=target)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,
                                                    random_state=27)

# Model: Decision Tree
df_results = pd.DataFrame(data=[])

# Without restrictions
# hyperparams: 'tree_depth': 19, 'tree_nodes': 1387, 'tree_leaves': 694
# results: 'mae': 4.3981, 'rmse': 6.3464, 'r2_score': 0.8462, 'pearson_r': 0.9206 

params_grid = dict()
params_grid["max_depth"] = [5, 6, 7, 8, 9, 10]
params_grid["max_leaf_nodes"] = [50, 75, 100, 150, 200, 250, 300]
params_grid["min_samples_split"] = [0.01, 0.025, 0.05, 0.075, 0.10]

params_keys = list(params_grid.keys())
params_comb = list(itertools.product(*params_grid.values()))

for i, hyper_params in enumerate(params_comb):
    md, mln, mss = hyper_params
    
    y_pred, params = regr_decisiontree(x_train, x_test, y_train,
                                       max_depth=md,
                                       max_leaf_nodes=mln,
                                       min_samples_split=mss,
                                       random_state=314,
                                       showfig=False, savefig=savefig)
    
    results = regr_metrics(y_test, y_pred)

    # Store Hyperparameters
    for key, value in zip(params_keys, hyper_params):
        df_results.loc[i, key] = value

    # Store Results
    for key, value in results.items():
        df_results.loc[i, key] = value


    
"""
plot_lineduo(x1=df_results.index, y1=df_results["mae"], label1="MAE",
             y2=df_results["rmse"], label2="RMSE", xlabel="max_leaf_nodes",
             title = f"Concrete Strength - Decision Tree - max_leaf_nodes")    

"""              
# Scout theme: "Always leave the campsite cleaner than you found it"
organize_report()
