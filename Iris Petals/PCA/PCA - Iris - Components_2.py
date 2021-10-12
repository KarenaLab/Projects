
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Program ---------------------------------------------------------------

print("\n ****  PCA Analysis - Iris Dataset  **** \n")

Filename = "iris.csv"
DF = pd.read_csv(Filename, sep= ",")

DF_Rows = DF.shape[0]
DF_Cols = DF.shape[1]
DF_Columns = DF.columns


# Setting Target

Target = "species"

Data_X = DF.drop(columns = [Target])
Data_Y = DF[Target]
Target_List = Data_Y.unique()
Target_Size = Data_Y.nunique()


# Standard Normalizing

Data_X_Columns = Data_X.columns

Scale = StandardScaler()
Transform = Scale.fit_transform(Data_X)
Data_X = pd.DataFrame(Transform, columns= Data_X_Columns)


# PCA Analysis

Components = 2
Comp_List = []

i = 1
while(i <= Components):

    Var_Name = "PC" + str(i)
    Comp_List.append(Var_Name)

    i = i+1
    

PCA = PCA(n_components= Components)
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

PCA.fit(Data_X)
PCA_X = PCA.transform(Data_X)

New_DF = pd.DataFrame(PCA_X, columns= Comp_List)
New_DF[Target] = Data_Y

PC1_Explain = PCA.explained_variance_ratio_[0]
PC2_Explain = PCA.explained_variance_ratio_[1]

print(f" >    PC1 Explain: {PC1_Explain*100:.3f}%")
print(f" >    PC2 Explain: {PC2_Explain*100:.3f}%")
print(f' > "Lost" of Info: {(1 - PC1_Explain - PC2_Explain)*100:.3f}%\n')


# Plotting

plt.style.use("ekc")

Alpha = 0.7
Color_List = ["navy", "darkred", "orange", "drakgreen"]


fig = plt.figure()
grd = fig.add_gridspec(nrows= 2, ncols= 2)

ax0 = fig.add_subplot(grd[0, 0])
ax1 = fig.add_subplot(grd[0, 1])
ax2 = fig.add_subplot(grd[1, 0])
ax3 = fig.add_subplot(grd[1, 1])

Title = f"PCA Analysis - Iris Petals [Components= {Components}]"
fig.suptitle(Title, fontsize= 14)


i = 0
No_Bins = int(np.sqrt(DF_Rows) + 0.5)

while(i < Target_Size):

    Group = Target_List[i]
    Color = Color_List[i]

    Data_PC1 = New_DF.groupby(Target).get_group(Group)["PC1"]
    Data_PC2 = New_DF.groupby(Target).get_group(Group)["PC2"]

    ax0.hist(Data_PC1, bins= No_Bins, color= Color, alpha= Alpha, label= Group)
    ax3.hist(Data_PC2, bins= No_Bins, color= Color, alpha= Alpha, label= Group)

    ax1.scatter(Data_PC2, Data_PC1, color= Color, alpha= Alpha, edgecolor= "white", label= Group)
    ax2.scatter(Data_PC1, Data_PC2, color= Color, alpha= Alpha, edgecolor= "white", label= Group)

    i = i+1


ax0.set_ylabel("PC1", fontweight= "bold", loc= "center")
ax2.set_ylabel("PC2", fontweight= "bold", loc= "center")
ax2.set_xlabel("PC1", fontweight= "bold", loc= "center")
ax3.set_xlabel("PC2", fontweight= "bold", loc= "center")

plt.legend(fontsize= 8)

plt.savefig(Title, dpi= 240)
plt.show()


# Closing

print(" * \n")

