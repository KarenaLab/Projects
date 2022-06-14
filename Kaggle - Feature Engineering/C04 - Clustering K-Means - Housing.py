
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Definitions

Seed = 27
np.random.seed(Seed)

plt.style.use("ekc")


# Program

print("\n ****  Kaggle - Feature Engineering - Clustering with K-Means  ****\n")


DF = pd.read_csv("housing.csv", sep= ",")


# K-Means = UNsupervised

Data_X = DF.loc[:, ["MedInc", "Latitude", "Longitude"]]

for n in range(2, 7):

    K_Means = KMeans(n_clusters= n)

    Data_X["Cluster"] = K_Means.fit_predict(Data_X)
    Data_X["Cluster"] = Data_X["Cluster"].astype("category")

    Colors = {0: "navy", 1: "darkred", 2: "orange", 3: "darkgreen", 4:"purple", 5:"lightblue"}


    # Plotting

    fig = plt.figure()

    Title = f"K-Means Clustering [{n}] - Housing"
    fig.suptitle(Title, fontsize= 14)

    plt.scatter(Data_X["Longitude"], Data_X["Latitude"],
                c= Data_X["Cluster"].map(Colors), s= 4, alpha= 0.5)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    #plt.savefig(Title, dpi= 240)
    plt.show()


# Closing

print("\n * \n")

