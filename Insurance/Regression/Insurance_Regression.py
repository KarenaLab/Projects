
# Libraries
import os
import sys
import time

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVR

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Personal modules
sys.path.append(r"c:\python_modules")

from color_source import *
from plot_histogram import *
from plot_heatmap import *
from one_hot_encoding import *
from split_trainval_test import *
from metrics_regression import *


# Setup/Config
path_main = os.getcwd()
path_report = os.path.join(path_main, "Report")


# Functions
target = "charges"
cols_numeric = ["age", "bmi", "charges"]
cols_categoric = ["sex", "children", "smoker", "region"]

def dict_to_list(dictionary):

    new_list = []
    for key in dictionary.keys():
        value = dictionary[key]
        temp = [key, value]
        new_list.append(temp)


    return new_list
        


# Program --------------------------------------------------------------------
filename = "insurance.csv"
df = pd.read_csv(filename, sep=",", encoding="utf-8")
df = one_hot_encoding(df, columns=cols_categoric)

df_trainval, df_test = split_trainval_test(df, train_size=.8, seed=18)
# TrainVal = 80% of dataset for Train and Validation,
#     Test = 20% of dataset for final Test. 


regr = LinearSVR()
epsilon_list = [0.1, 0.2, 0.5, 0.9, 1]
results = pd.DataFrame(data=[])

n_folds = 5
# 20% for test -> n_folds = 1 / 0.2 = 5

kf = KFold(n_splits=n_folds, shuffle=True)
for i, (train_index, val_index) in enumerate(kf.split(df_trainval)):
    df_train = df_trainval.loc[train_index, :].reset_index(drop=True)
    df_val = df_trainval.loc[val_index, :].reset_index(drop=True)

    x_train, y_train = df_train.drop(columns=[target]), df_train[target]
    x_val, y_val = df_val.drop(columns=[target]), df_val[target]

    scaler = StandardScaler()
    scaler = scaler.fit(x_train)
    scx_train = scaler.transform(x_train)
    scx_val = scaler.transform(x_val)

    for e in epsilon_list:
        regr.epsilon = e
        print(f" > regr.epsilon = {e}")
        
        regr.fit(x_train, y_train)
        y_pred = regr.predict(x_val)

        metrics = regr_metrics(y_val, y_pred)


    


    



    
