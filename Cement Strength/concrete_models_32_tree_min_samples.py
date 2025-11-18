# [P504] Cement compressive strength
# Performs EDA (Exploratory Data Analysis)

# Libraries
import os

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

    return results

                   
# Setup/Config
savefig = False

    
# Program --------------------------------------------------------------
df = load_dataset()
target = "compressive_strength_mpa"

# Data Split
x, y = split_target(df, target=target)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,
                                                    random_state=27)

# Model without restrictions
# hyperparams: 'tree_depth': 19, 'tree_nodes': 1387, 'tree_leaves': 694
# results: 'mae': 4.3981, 'rmse': 6.3464, 'r2_score': 0.8462, 'pearson_r': 0.9206 


# Model: Decision Tree
df_results = pd.DataFrame(data=[])
for i in range(1, 25+1):
    split_pct = np.round(i/100, decimals=2)
    y_pred, params = regr_decisiontree(x_train, x_test, y_train,
                                       min_samples_split=split_pct,
                                       random_state=314,
                                       showfig=False, savefig=savefig)
    
    results = regr_metrics(y_test, y_pred)

    for key, value in results.items():
        df_results.loc[split_pct, key] = value


df_results = df_results.sort_index(ascending=False)

plot_lineduo(x1=df_results.index, y1=df_results["mae"], label1="MAE",
             y2=df_results["rmse"], label2="RMSE", xlabel="min_samples_split (%)",
             title = f"Concrete Strength - Decision Tree - min_samples_split")    

                 
# Scout theme: "Always leave the campsite cleaner than you found it"
organize_report()
