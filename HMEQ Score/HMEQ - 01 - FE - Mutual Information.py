
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_regression

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



# Definitions -----------------------------------------------------------

Seed = 27
np.random.seed(Seed)

plt.style.use("ekc")


def Rounder(x):

    x = np.round(x, decimals= 0)
    return x



# Program ---------------------------------------------------------------

print("\n ****  HMEQ - Credit Score  **** \n")


# Getting Information from .csv

Filename = "hmeq.csv"

DF = pd.read_csv(Filename, sep= ",")
DF_SafeCopy = DF.copy()

DF_Rows = DF.shape[0]
DF_Cols = DF.shape[1]
DF_Columns = DF.columns


# NaNs Report

Total = 0

for col in DF_Columns:

    Data = DF[col]
    NaN_Count = Data.isnull().sum()
    Total = Total + NaN_Count

    print(f" > {col}: NaNs = {NaN_Count} ({(NaN_Count/DF_Rows)*100:.1f}%)")


print(f"\n >>> Total of NaNs = {Total}\n")


# Preparation for KNN: Changing Text columns into Numbers/Cetegoric

for col in DF.select_dtypes("object"):

    DF[col], trash = DF[col].factorize()
    DF[col] = DF[col].astype(dtype= "int64")

               

# KNN Imputation

Target = "BAD"

Data_X = DF.drop(columns= [Target])
Data_Y = DF[Target]

Data_X_Columns = Data_X.columns


# KNN Imputer

imputer = KNNImputer(n_neighbors= 5,
                     weights= "uniform", metric= "nan_euclidean")

DF_Transform = imputer.fit_transform(Data_X, Data_Y)
DF_Transform = pd.DataFrame(data= DF_Transform, columns= Data_X_Columns)


# Mutual Information

Data_X = DF_Transform.copy()
Data_Y = Data_Y.astype(dtype= "int64")


# Label encoding for categoricals

for col in Data_X.select_dtypes("object"):

    Data_X[col], trash = Data_X[col].factorize()
    Data_X[col] = Data_X[col].astype(dtype= "int64")



# Mutual Information Check

Discrete_Features = (Data_X.dtypes == "int64")

MI_Scores = mutual_info_regression(Data_X, Data_Y,
                                   discrete_features= Discrete_Features)

MI_Scores = pd.Series(MI_Scores, name= "MI Scores", index= Data_X.columns)
MI_Scores = MI_Scores.sort_values(ascending= False)


# Plotting MI Scores

Scores = MI_Scores.sort_values(ascending= True)
Width = np.arange(0, len(Scores))
Ticks = list(Scores.index)

fig = plt.figure()

Title = "Feature Engineering - Mutual Info - HMEQ"
fig.suptitle(Title, fontsize= 14)

plt.barh(Width, Scores, color= "navy")
plt.yticks(ticks= Width, labels= Ticks)

plt.xlabel("Mutual Information Score")

plt.grid(b= False, axis= "y")
#plt.savefig(Title, dpi= 240)
plt.show()

