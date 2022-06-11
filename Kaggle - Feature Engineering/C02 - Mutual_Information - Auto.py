
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_regression

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Definitions ----------------------------------------------------------

Seed = 27
np.random.seed(Seed)

plt.style.use("ekc")


# Program --------------------------------------------------------------

print("")
print(" ****  Kaggle Course - Feature Engineering - Class 02  **** ")
print(" ****              Mutual Information                  **** \n")


DF = pd.read_csv("autos.csv", sep= ",")


Target = "price"

Data_X = DF.copy()
Data_X = DF.drop(columns= [Target])

Data_Y = DF[Target]
Data_Y = Data_Y.astype(dtype= int)


# Label encoding for categoricals

for col in Data_X.select_dtypes("object"):

    Data_X[col], trash = Data_X[col].factorize()
    Data_X[col] = Data_X[col].astype(dtype= int)



# Mutual Information Check

Discrete_Features = (Data_X.dtypes == int)

MI_Scores = mutual_info_regression(Data_X, Data_Y,
                                   discrete_features= Discrete_Features)

MI_Scores = pd.Series(MI_Scores, name= "MI Scores", index= Data_X.columns)
MI_Scores = MI_Scores.sort_values(ascending= False)


# Plotting MI Scores

Scores = MI_Scores.sort_values(ascending= True)
Width = np.arange(0, len(Scores))
Ticks = list(Scores.index)

fig = plt.figure()

Title = "Feature Engineering - Mutual Info - Autos"
fig.suptitle(Title, fontsize= 14)

plt.barh(Width, Scores, color= "navy")
plt.yticks(ticks= Width, labels= Ticks)

plt.xlabel("Mutual Information Score")

plt.grid(b= False, axis= "y")
#plt.savefig(Title, dpi= 240)
plt.show()



# Sources ---------------------------------------------------------------

# https://pandas.pydata.org/docs/reference/api/pandas.factorize.html
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html









