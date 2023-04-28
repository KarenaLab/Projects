
# Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from Financial_Distress_Tools_v01 import *

# Setup/Config
seed = 42


# Program ------------------------------------------------------------
print("\n ****  Financial Distress - Linear Regression  ****\n")

filename = "Financial-Distress.csv"
df = pd.read_csv(filename, sep=",", encoding="utf-8")


# Dataset adjusts
df["x80"] = df["x80"].apply(lambda x: str(x))
df = remove_columns(df, columns=["Company", "Time"])

# Categoric 
df = one_hot_encoding(df, column="x80")
df_trainvalid, df_test = train_test_split(df, train_size=80, seed=seed, verbose=True)


# Model --------------------------------------------------------------
n_splits = 5
kf_shuffle = True

cols_discrete = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
                 "x9", "x10", "x11", "x12", "x13", "x14", "x15",
                 "x16", "x17", "x18", "x19", "x20", "x21", "x22",
                 "x23", "x24", "x25", "x26", "x27", "x28", "x29",
                 "x30", "x31", "x32", "x33", "x34", "x35", "x36",
                 "x37", "x38", "x39", "x40", "x41", "x42", "x43",
                 "x44", "x45", "x46", "x47", "x48", "x49", "x50",
                 "x51", "x52", "x53", "x54", "x55", "x56", "x57",
                 "x58", "x59", "x60", "x61", "x62", "x63", "x64",
                 "x65", "x66", "x67", "x68", "x69", "x70", "x71",
                 "x72", "x73", "x74", "x75", "x76", "x77", "x78",
                 "x79", "x81", "x82", "x83"]


# Split: trainval -> train and validation
kf = KFold(n_splits=n_splits, shuffle=kf_shuffle)

results = []
for n_fold, (train_index, valid_index) in enumerate(kf.split(df_trainvalid), start=1):
    print(f" Fold {n_fold}")   
    df_train = df_trainvalid.loc[train_index, :]
    df_valid = df_trainvalid.loc[valid_index, :]

    # Split: Variables and Target
    target = "Financial Distress"
    x_train, y_train = target_split(df_train, target=target)
    x_valid, y_valid = target_split(df_valid, target=target)

    # Normalization: Min-Max
    x_train, param_minmax = scaler_standard(x_train, columns=cols_discrete)
    x_valid, _ = scaler_standard(x_valid, param=param_minmax)
        
    # Model: Linear Regression
    y_pred, param_model, model_intercept, model_coef = model_linear_ridge(x_train, y_train, x_valid)
    metrics = metrics_reg(y_valid, y_pred, metrics="all")
    results.append([n_fold, metrics])


