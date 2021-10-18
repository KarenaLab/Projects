
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

Var_Study = ["petal_width"]
Target = "species"

Data_X = DF[Var_Study]
Data_Y = DF[Target]


Species_Classification = { "setosa": 0,
                           "versicolor": 0,
                           "virginica": 1 }

Data_Y = Data_Y.map(Species_Classification)


# Logistic Regression Model

Logit = LogisticRegression()
Logit.fit(Data_X, Data_Y)

Y_Fit = Logit.predict_proba(Data_X)


X_New = np.linspace(start= 0, stop= 3, num= 1000)
X_New = X_New.reshape(-1, 1)

Y_Pred = Logit.predict_proba(X_New)


DF_Plot = pd.DataFrame()
DF_Plot["petal_width"] = DF["petal_width"]
DF_Plot["virginica"] = Data_Y
DF_Plot["proba_1"] = Y_Fit[:, 1]

Color_Map = { 0: "royalblue",
              1: "darkgreen" }

DF_Plot["color"] = DF_Plot["virginica"].map(Color_Map)


# Plotting

plt.style.use("ekc")

fig = plt.figure()

Title = "Iris Petals - Logit Study for Petals"
plt.suptitle(Title, fontsize= 14)

plt.plot(X_New, Y_Pred[:, 1], color= "darkgreen", label= "Virginica", zorder = 10)
plt.plot(X_New, Y_Pred[:, 0], color= "royalblue", linestyle= "--", label= "NOT Virginica", zorder= 11)

plt.scatter(DF_Plot["petal_width"], DF_Plot["proba_1"], color= DF_Plot["color"], alpha= 0.6, zorder= 12)

plt.xlabel("Petal Width (cm)")
plt.ylabel("Probability")



plt.show()


# Closing

print(" * \n")

    



