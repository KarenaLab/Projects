
# Iris Petals

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt


print("\n ****  Iris Petals - LDA - Linear Discriminant Analysis  **** \n")


DF = pd.read_csv("iris.csv", sep= ",")

Target = "species"

Data_X = DF.drop(columns = [Target])
Data_Y = DF[Target]


# LDA Reduction Model

print(" Important: n_components cannot be larger than")
print("                min(n_features, n_classes - 1). \n")


for n in range(1, 3):

    LDA = LinearDiscriminantAnalysis(n_components= n)
    LDA_Model = LDA.fit(Data_X, Data_Y)

    LDA_X = LDA_Model.transform(Data_X)

    VarianceRatio = LDA.explained_variance_ratio_
    VarianceRatio = np.sum(VarianceRatio)
    
    print(f" > Explained Variance Ratio with {n} components: {VarianceRatio:.5f}")


print("")


# Closing

print(" * \n")



