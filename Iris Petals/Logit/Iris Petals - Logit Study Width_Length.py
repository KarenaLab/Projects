
# Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow
# Aurélien Géron

# Cap.04 - Training Models - p.148~154


import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



# Program

print("\n ****  Iris Petals - Logit (Geron p.150)  **** \n")


Filename = "iris.csv"
DF = pd.read_csv(Filename, sep= ",")


# Data Preparation

Var_Study = ["petal_length", "petal_width"]
Target = "species"

Data_X = DF[Var_Study]

Species_Classification = { "setosa": 0,
                           "versicolor": 0,
                           "virginica": 1 }

Data_Y = DF[Target].map(Species_Classification)


# Logistic Regression Model


Color_Map = {0: "royalblue",
             1: "darkgreen" }

Color = Data_Y.map(Color_Map)


# Plotting

plt.style.use("ekc")

fig = plt.figure()

Title = "Iris Petals - Logit Linear Decision Boundary"
plt.suptitle(Title, fontsize= 14)

plt.scatter(Data_X["petal_length"], Data_X["petal_width"], c= Color)

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")




plt.show()


# Closing

print(" * \n")

    



