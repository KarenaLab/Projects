
#   Project: 123 - Height and Weight Prediction
#  Filename: Height_Weight_Prediction.py
#   Creator: EKChikui, 
#      Date: Sept 28th, 2021
#   Version: 01
#     Descr: 
#

# Version Control -------------------------------------------------------

# 01 - 28/09/2021 - Launch Version
# 02 - 


# Libraries -------------------------------------------------------------

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from HistBoxInfoNormal_v16 import *
from ScatterHistLinReg_v05 import *


# Definitions -----------------------------------------------------------

Seed = 27
np.random.seed(Seed)


# Setup -----------------------------------------------------------------



# MAIN Program ----------------------------------------------------------

print("\n ****  Heights and Weights Prediction  **** \n")

# Data Acquisition

Filename = "HeightsWeights.csv"
DF = pd.read_csv(Filename, sep= ",")


# Data Preparation

DF = DF.drop(columns= ["Index"])

Rename_Dict = { "Height(Inches)": "height",
                "Weight(Pounds)": "weight" }

DF = DF.rename(columns= Rename_Dict)

DF["height"] = DF["height"] * 2.54          # Unit = centimeters
DF["weight"] = DF["weight"] * 0.453592      # Unit = kg


# Data Wrangling

DF_Rows = DF.shape[0]
DF_Cols = DF.shape[1]
DF_Columns = DF.columns


for col in DF_Columns:

    Data = DF[col]
    NaN_Count = Data.isnull().sum()

    print(f" > {col}: NaNs = {NaN_Count} ({(NaN_Count/DF_Rows)*100:.1f}%)")


print("")


# Plottings

fig = plt.figure(figsize= (8, 4.5))

Title = "Height Weight Study - 01 - Scatter"
fig.suptitle(Title, fontsize= 14)

Data_X = DF["height"]
Data_Y = DF["weight"]

plt.scatter(Data_X, Data_Y, alpha= 0.5, edgecolor= "white", zorder= 6)

plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")

plt.grid(axis= "both", color= "lightgrey", linestyle= "--", linewidth= 0.5, zorder= 5)


#plt.savefig(Title, dpi= 240)
plt.show()


Title = f"Height Weight Study - 02 - Scatter Detailed"
ScatterHistLinReg(Title, Data_X, Data_Y, savefig= False)


for col in DF_Columns:

    Title = f"Height Weight Study - 03 - Histogram [{col}]"
    Data = DF[col]

    HistBoxInfoNormal(Title, Data, savefig= False)



# Sources ---------------------------------------------------------------

# https://blog.cambridgespark.com/six-data-science-projects-to-top-up-your-skills-and-knowledge-2f073fd80f55
# http://socr.ucla.edu/docs/resources/SOCR_Data/SOCR_Data_Dinov_020108_HeightsWeights.html
# https://www3.nd.edu/~steve/computing_with_data/2_Motivation/motivate_ht_wt.html
#
#
