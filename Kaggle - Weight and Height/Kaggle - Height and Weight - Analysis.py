
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from ScatterHistLinReg_v05 import *
from HistBoxInfoNormal_v16 import *


# Program --------------------------------------------------------------

print("\n ****  Kaggle Height and Weight Dataset  **** \n")

Filename = "Kaggle_Weight_Height.csv"
DF = pd.read_csv(Filename, sep= ",")
DF_SafeCopy = DF.copy()

DF_Rows = DF.shape[0]
DF_Cols = DF.shape[1]
DF_Columns = DF.columns


DF["Height"] = DF["Height"] * 2.54          # inches to cm*
DF["Weight"] = DF["Weight"] * 0.453592      # pounds to kg*

Gender_List = ["Male", "Female"]
Variable_List = ["Height", "Weight"]


SaveReport = False

# Scatter Plot

for gender in Gender_List:

    Data = DF.groupby("Gender").get_group(gender)

    Data_X = Data["Height"]
    Data_Y = Data["Weight"]

    Title = f"Kaggle Weight Height - 01 - {gender}"

    ScatterHistLinReg(Title, Data_X, Data_Y, savefig= SaveReport)


# Histogram

for gender in Gender_List:

    for variable in Variable_List:

        Data = DF.groupby("Gender").get_group(gender)
        Data = Data[variable]

        Title = f"Kaggle Weight Height - 02 - {gender} - {variable}"

        HistBoxInfoNormal(Title, Data, savefig= SaveReport)
   
