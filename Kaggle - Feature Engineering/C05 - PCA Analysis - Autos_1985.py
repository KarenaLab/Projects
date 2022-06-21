
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Program ---------------------------------------------------------------

print("\n ****  PCA Analysis - 1985 Automobiles Dataset  **** \n")


Filename = "autos_1985.csv"
DF = pd.read_csv(Filename, sep= ",")


# Pre-Processing

Target = "price"

DF_Numeric = [ "symboling", "num_of_doors", "wheel_base", "length",
               "width", "height", "curb_weight", "num_of_cylinders",
               "engine_size", "bore", "stroke", "compression_ratio",
               "horsepower", "peak_rpm", "city_mpg", "highway_mpg",
               "price" ]

Data_X = DF[DF_Numeric]
Data_Y = DF[Target]


Data_X_Columns = Data_X.columns

Scale = MinMaxScaler()
# StandardScaler or MinMaxScaler

Transform = Scale.fit_transform(Data_X)
Data_X = pd.DataFrame(Transform, columns= Data_X_Columns)


# PCA Analysis

No_Comp_List = []
Var_Ratio_List = []

Max_Comp = len(Data_X_Columns)+1

for Comp in range(2, Max_Comp):

    # Tag for New DF with Principal Components (PCx)
    
    Comp_List = []

    i = 1
    while(i <= Comp):

        Var_Name = "PC" + str(i)
        Comp_List.append(Var_Name)

        i = i+1


    # Model

    pca = PCA(n_components= Comp)

    pca.fit(Data_X)
    pca_X = pca.transform(Data_X)

    New_DF = pd.DataFrame(pca_X, columns= Comp_List)
    New_DF[Target] = Data_Y

    Var_Ratio = pca.explained_variance_ratio_
    Var_Ratio = np.round(np.sum(Var_Ratio)*100, decimals= 3)

    No_Comp_List.append(Comp)
    Var_Ratio_List.append(Var_Ratio)
    
    print(f"Principal Components: {Comp} - VarRatio = {Var_Ratio}")


# Plotting

plt.style.use("ekc")

fig = plt.figure()

Title = "PCA Analysis - Kaggle 1985 Autos"
plt.suptitle(Title, fontsize= 14)

plt.plot(No_Comp_List, Var_Ratio_List, color= "navy")
plt.axhline(y= 95, color= "orange", linestyle= "--", linewidth= 0.5)
plt.axhline(y= 99, color= "darkred", linestyle= "--", linewidth= 0.5)

plt.xlabel("No Principal Components")
plt.ylabel("% Cumulative Variance")

plt.savefig(Title, dpi= 240)
plt.show()


# Closing

print("\n * \n")


