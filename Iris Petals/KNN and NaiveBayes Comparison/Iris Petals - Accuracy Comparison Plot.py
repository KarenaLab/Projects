
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Program ---------------------------------------------------------------

print("\n ****  Iris Petals - KNN versus Naive Bayes  **** \n")


Filename = "iris_KNN_and_NB.csv"
DF = pd.read_csv(Filename, sep= ",")


# Data Preparation

Data_KNN = DF["species_KNN"]
Data_NB = DF["species_NB"]

DF = DF.drop(columns= ["species_KNN", "species_NB"])


DF_Rows = DF.shape[0]
DF_Cols = DF.shape[1]
DF_Columns = DF.columns


# PCA Preparing

Target = "species"

Data_X = DF.drop(columns= [Target])
Data_Y = DF[Target]


Target_List = Data_Y.unique()
Target_Size = Data_Y.nunique()


Data_X_Columns = Data_X.columns

Scale = MinMaxScaler()
Transform = Scale.fit_transform(Data_X)

Data_X = pd.DataFrame(Transform, columns= Data_X_Columns)


# PCA Analysis

Comp_List = ["PC1", "PC2"]

pca = PCA(n_components= 2)

pca.fit(Data_X)
pca_X = pca.transform(Data_X)

New_DF = pd.DataFrame(pca_X, columns= Comp_List)
New_DF[Target] = Data_Y

New_DF["species_KNN"] = Data_KNN
New_DF["species_NB"] = Data_NB


# Color Mapping

Strategy_List = ["KNN", "NB"]

for Strategy in Strategy_List:

    col_in = "species_" + Strategy
    col_out = Strategy + "_color"      

    Color_Map = []

    i = 0
    while(i < DF_Rows):

        Ref = New_DF.loc[i, "species"]
        Comp = New_DF.loc[i, col_in]

        if (Ref == Comp):

            if(Ref == "setosa"):
                Color = "cornflowerblue"

            if(Ref == "versicolor"):
                Color = "yellowgreen"

            if(Ref == "virginica"):
                Color = "wheat"

        else:
            Color = "red"
            print(f"{Strategy} = [{i}] = Error. Right = {Ref}, Model = {Comp}")


        Color_Map.append(Color)
        i = i+1


    New_DF[col_out] = Color_Map
    print("")


# Plotting

plt.style.use("ekc")

fig = plt.figure()
grd = gridspec.GridSpec(nrows= 1, ncols= 2)

ax0 = fig.add_subplot(grd[0, 0])
ax1 = fig.add_subplot(grd[0, 1])


Title = "Iris Petals - Accuracy Comparison"
plt.suptitle(Title, fontsize= 14)

Data_X = New_DF["PC1"]
Data_Y = New_DF["PC2"]

KNN_Color = New_DF["KNN_color"]
NB_Color = New_DF["NB_color"]

ax0.scatter(Data_X, Data_Y, c= KNN_Color, alpha= 0.7)
ax0.set_title("KNN Accuracy")

ax0.set_xlabel("PC1", fontweight= "bold", loc= "center")
ax0.set_ylabel("PC2", fontweight= "bold", loc= "center")

ax1.scatter(Data_X, Data_Y, c= NB_Color, alpha= 0.7)
ax1.set_title("Categorical Naive Bayes Accuracy")

ax1.set_xlabel("PC1", fontweight= "bold", loc= "center")

plt.savefig(Title, dpi= 240)
plt.show()



# Closing

print(" * \n")



    

    








