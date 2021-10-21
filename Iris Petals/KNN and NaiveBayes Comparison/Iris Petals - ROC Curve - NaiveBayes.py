
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import CategoricalNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



# Program --------------------------------------------------------------

print("\n ****  Iris Petals - Categorical Naive Bayes Proba  **** \n")


Filename = "iris_test.csv"
DF = pd.read_csv(Filename, sep= ",")


Target = "species"
Seed = 27


# Adjusting Target Info (Cat. Text to Cat. Numeric)

Species_Dict = { "setosa": 0,
                 "versicolor": 1,
                 "virginica": 2 }

DF[Target] = DF[Target].map(Species_Dict)



# Categorical Naive Bayes Solution (Best Sample= 48*) ------------------

Sample = 48

DF_Ready = DF.dropna().copy()
DF_Missing = DF[DF[Target].isna()].copy()

Data_Ready_X = DF_Ready.drop(columns= [Target])     # Feature Matrix
Data_Ready_Y = DF_Ready[Target]                     # Response Vector


Sample = Sample/100

X_Train, X_Test, Y_Train, Y_Test = train_test_split(Data_Ready_X, Data_Ready_Y, test_size= Sample, random_state= Seed)

Categorical_NaiveBayes = CategoricalNB()
Categorical_NaiveBayes.fit(X_Train, Y_Train)

Y_Preds = Categorical_NaiveBayes.predict_proba(X_Test)


# Plotting

plt.style.use("ekc")
fig = plt.figure()

Title = "Iris Petals - Categorical Naive Bayes Proba"
fig.suptitle(Title, fontsize= 14)

plt.scatter(range(0, Y_Preds.shape[0]), Y_Preds[:, 0], color= "navy", label= "setosa")
plt.scatter(range(0, Y_Preds.shape[0]), Y_Preds[:, 1], color= "darkred", label= "versicolor")
plt.scatter(range(0, Y_Preds.shape[0]), Y_Preds[:, 2], color= "orange", label= "virginica")

plt.axhline(y= 0.80, color= "black", linestyle= "--", linewidth= 0.5)
plt.axhline(y= 0.95, color= "black", linestyle= "--", linewidth= 0.5)
plt.ylim(bottom= 0.50, top= 1.05)

plt.xlabel("Species = NaN")
plt.ylabel("Probability")

plt.legend()
#plt.savefig(Title, dpi= 240)

plt.show()


# Closing

print("\n * \n")

