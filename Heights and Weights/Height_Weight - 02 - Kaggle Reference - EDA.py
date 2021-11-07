
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

Filename = "Kaggle_Weight_Height.csv"
DF = pd.read_csv(Filename, sep= ",")

DF_Rows = DF.shape[0]
DF_Cols = DF.shape[1]
DF_Columns = DF.columns


# Data Preparation

DF["Height"] = DF["Height"] * 2.54          # Unit = centimeters
DF["Weight"] = DF["Weight"] * 0.453592      # Unit = kg


# Feature Engineering

IMC_List = []
Ratio_List = []

i = 0
while(i < DF_Rows):

    height = DF.loc[i, "Height"]
    weight = DF.loc[i, "Weight"]

    IMC = weight/((height/100)**2)
    Ratio = weight/height

    IMC_List.append(IMC)
    Ratio_List.append(Ratio)

    i = i+1


DF["IMC"] = IMC_List
DF["Ratio_WH"] = Ratio_List


# Categorical Data

Gender_Dict = { "Male": 0,
                "Female": 1 }

DF["Gender"] = DF["Gender"].map(Gender_Dict)


# Separated Analysis by: Gender

Gender_List = [0, 1]
Variable_List = ["Height", "Weight", "IMC", "Ratio_WH"]

SaveReport = True


for Gender in Gender_List:

    if(Gender == 0):
        Gender_Text = "Male"

    else:
        Gender_Text = "Female"
    

    for Variable in Variable_List:

        Data = DF.groupby(by= "Gender").get_group(Gender)[Variable]
        Title = "Weight Height - " + Gender_Text + " - " + Variable

        HistBoxInfoNormal(Title, Data, savefig= SaveReport)
        



# Sources ---------------------------------------------------------------

# https://blog.cambridgespark.com/six-data-science-projects-to-top-up-your-skills-and-knowledge-2f073fd80f55
# http://socr.ucla.edu/docs/resources/SOCR_Data/SOCR_Data_Dinov_020108_HeightsWeights.html
# https://www3.nd.edu/~steve/computing_with_data/2_Motivation/motivate_ht_wt.html
# https://ourworldindata.org/human-height
# 
#
