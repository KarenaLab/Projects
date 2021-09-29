
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# Program --------------------------------------------------------------

print("\n ****  Kaggle Height and Weight Dataset  **** \n")

Filename = "Kaggle_Weight_Height.csv"
DF = pd.read_csv(Filename, sep= ",")
DF_SafeCopy = DF.copy()

DF_Rows = DF.shape[0]
DF_Cols = DF.shape[1]
DF_Columns = DF.columns


DF["Height"] = DF["Height"] * 2.54          # inches to cm*
DF["Weight"] = DF["Weight"] * 0.453592      # pounds to kg*

Gender_List = ["Male", "Female"]
Variable_List = ["Height", "Weight"]


# Logit Model

# Variables

Size = 0.2
Seed = 27
Threshold = 0.5


# Preparing String Columns

Data = DF["Gender"]
Gender_Map = {"Male": 0, "Female": 1}
Data = Data.map(Gender_Map)

DF["Gender"] = Data


# Train/Test Preparation

Target = "Gender"

Data_X = DF.drop(columns= [Target])
Data_Y = DF[Target]


# Logit

Train_R2_Square_List = []
Train_Accuracy_List = []
Train_MAE_List = []

Test_R2_Square_List = []
Test_Accuracy_List = []
Test_MAE_List = []

Size_List = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

for Size in Size_List:

    print(f" >>> Test Sample Size = {Size}%")

    Size = Size/100

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(Data_X, Data_Y, test_size= Size, random_state= Seed)

    Logit = LogisticRegression()
    Logit.fit(X_Train, Y_Train)


    # TRAIN Metrics

    Y_Proba_Train = Logit.predict_proba(X_Train)

    Y_Predict_Train = []

    for x in Y_Proba_Train:

        get = x[1]

        if(get >= Threshold):
            Solution = 1

        else:
            Solution = 0


        Y_Predict_Train.append(Solution)


    Y_Predict_Train = np.array(Y_Predict_Train)


    Train_R2_Square = metrics.r2_score(Y_Train, Y_Predict_Train)
    Train_MSE = metrics.mean_squared_error(Y_Train, Y_Predict_Train)
    Train_MAE = metrics.mean_absolute_error(Y_Train, Y_Predict_Train)
    Train_Accuracy = metrics.accuracy_score(Y_Train, Y_Predict_Train)

    Train_R2_Square_List.append(Train_R2_Square)
    Train_MAE_List.append(Train_MAE)
    Train_Accuracy_List.append(Train_Accuracy)

    #print(f" > Train R2 Square: {Train_R2_Square:.5f}")
    #print(f" >  Train Accuracy: {Train_Accuracy:.5f}") 
    #print(f" >       Train MAE: {Train_MAE:.5f}\n")
    

    # TEST Metrics

    Y_Proba_Test = Logit.predict_proba(X_Test)

    Y_Predict_Test = []

    for x in Y_Proba_Test:

        get = x[1]

        if(get >= Threshold):
            Solution = 1

        else:
            Solution = 0


        Y_Predict_Test.append(Solution)


    Y_Predict_Test = np.array(Y_Predict_Test)


    Test_R2_Square = metrics.r2_score(Y_Test, Y_Predict_Test)
    Test_MSE = metrics.mean_squared_error(Y_Test, Y_Predict_Test)
    Test_MAE = metrics.mean_absolute_error(Y_Test, Y_Predict_Test)
    Test_Accuracy = metrics.accuracy_score(Y_Test, Y_Predict_Test)

    Test_R2_Square_List.append(Test_R2_Square)
    Test_MAE_List.append(Test_MAE)
    Test_Accuracy_List.append(Test_Accuracy)

    #print(f" > Test R2 Square: {Test_R2_Square:.5f}")
    #print(f" >  Test Accuracy: {Test_Accuracy:.5f}") 
    #print(f" >       Test MAE: {Test_MAE:.5f}\n")


# Plotting

plt.style.use("ekc")

fig = plt.figure()

Title = "Kaggle Height Weight - Logit - Cross Validation"
fig.suptitle(Title, fontsize= 14)

plt.plot(Size_List, Train_Accuracy_List, color= "navy", label= "Train")
plt.plot(Size_List, Test_Accuracy_List, color= "darkred", label= "Test")

plt.xticks(Size_List)
plt.xlabel("Test Size (%)")
plt.ylabel("Accuracy (%)")

plt.legend()

#plt.savefig(Title, dpi= 240)
plt.show()

print("")



