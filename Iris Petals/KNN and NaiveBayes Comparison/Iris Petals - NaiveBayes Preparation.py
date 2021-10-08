
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



# Program ---------------------------------------------------------------

print("\n ****  Iris Petals - KNN Imputation  **** \n")

Filename = "iris_test.csv"
DF = pd.read_csv(Filename, sep= ",")
DF_SafeCopy = DF.copy()


Target = "species"
Seed = 38


# Finding Best Parameters

DF = DF.dropna().reset_index(drop= True)

Species_Dict = { "setosa": 0,
                 "versicolor": 1,
                 "virginica": 2 }

DF[Target] = DF[Target].map(Species_Dict)


Data_X = DF.drop(columns= [Target])     # Feature Matrix
Data_Y = DF[Target]                     # Response Vector


Acc_Train_List = []
Acc_Test_List = []

# Train/Test

Sample_List = np.linspace(start= 25, stop= 55, num= 7)

for Sample in Sample_List:

    print(f" > Performing Categorical Naive Bayes for Sample = {int(Sample)}%")

    Sample = Sample/100
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(Data_X, Data_Y, test_size= Sample, random_state= Seed)

    Categorical_NaiveBayes = CategoricalNB()
    Categorical_NaiveBayes.fit(X_Train, Y_Train)

    Y_Pred_Train = Categorical_NaiveBayes.predict(X_Train)
    Y_Pred_Test = Categorical_NaiveBayes.predict(X_Test)

    # Metrics
    Acc_Train = metrics.accuracy_score(Y_Train, Y_Pred_Train)*100
    Acc_Test = metrics.accuracy_score(Y_Test, Y_Pred_Test)*100

    Acc_Train_List.append(Acc_Train)
    Acc_Test_List.append(Acc_Test)


# Plotting

plt.style.use("ekc")

fig = plt.figure()

Title = "Iris Petals - Gaussian Naive Bayes Best Sample Size"
fig.suptitle(Title, fontsize= 14)

plt.plot(Sample_List, Acc_Train_List, color= "navy", label= "Train")
plt.plot(Sample_List, Acc_Test_List, color= "darkred", label= "Test")

plt.axvline(x= 50, color= "black", linestyle= "--", linewidth= 0.5)

plt.xlabel("% Test Size")
plt.ylabel("% Accuracy")

plt.legend(loc= "lower right")
plt.savefig(Title, dpi= 240)
plt.show()


# Closing

print("\n * \n")
    




