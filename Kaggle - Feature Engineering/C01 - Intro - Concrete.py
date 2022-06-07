
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


# Program --------------------------------------------------------------

print("\n ****  Kaggle Course - Feature Engineering - Class 01  **** \n")


# Getting Information

DF = pd.read_csv("concrete.csv", sep= ",")

Target = "CompressiveStrength"

Data_X = DF.copy()
Data_X = DF.drop(columns= [Target])
Data_Y = DF[Target]


# Train and score baseline model

baseline = RandomForestRegressor(criterion= "mae", random_state= 0)
baseline_score = cross_val_score(baseline, Data_X, Data_Y,
                                 cv= 5, scoring= "neg_mean_absolute_error")

baseline_score = (-1) * baseline_score.mean()
print(f" > MAE Baseline Score: {baseline_score:.4}")


# Creating Synthetic Features (Feature Engineering)

Data_X["FCRatio"] = Data_X["FineAggregate"] / Data_X["CoarseAggregate"]
Data_X["AggCmtRatio"] = (Data_X["CoarseAggregate"] + Data_X["FineAggregate"]) / Data_X["Cement"]
Data_X["WtrCmtRatio"] = Data_X["Water"] / Data_X["Cement"]


model_01 = RandomForestRegressor(criterion= "mae", random_state= 0)
model_01_score = cross_val_score(model_01, Data_X, Data_Y,
                                 cv= 5, scoring= "neg_mean_absolute_error")

model_01_score = (-1) * model_01_score.mean()
print(f" > MAE Model 01 Score: {model_01_score:.4}")


# Closing

print("\n * \n")


# Sources ---------------------------------------------------------------

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
#
