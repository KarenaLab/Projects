# Name [P421] - Canada retail profit (Classification)


# Libraries
import os
import sys
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.stats as st

import matplotlib.pyplot as plt


# Personal modules
from canada_profit_tools import *

sys.path.append(r"c:\python_modules")
from plot_barv import plot_barv
from plot_histbox import plot_histbox

# Functions



# Setup/Config
savefig = False


# Program --------------------------------------------------------------
df = load_dataset()


for col in cat_cols():
    data = df[col].value_counts()
    x = np.array(data.index)
    y = list(data.values)
    plot_barv(x, height=y, title=f"ecommerce - Category - {col}", savefig=savefig)
    
for col in num_cols():
    data = df[col]
    plot_histbox(data, title=f"ecommerce - Histogram - {col}", savefig=savefig)



"""
# Numerical to Categoric
df["sales_cat"] = numeric_to_categoric(df["sales"], no_cat=10)
df = df.drop(columns=["sales"])

df["order_quantity_cat"] = numeric_to_categoric(df["order_quantity"], no_cat=10)
df = df.drop(columns=["order_quantity"])

# Train/Test split
x, y = split_target(df, target="profit")
x = pd.get_dummies(x, drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Model
regr = DecisionTreeRegressor()

# Hyperparameters


# Fit and predict
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
metrics = regr_metrics(y_test, y_pred)
print(metrics)
"""



# end

