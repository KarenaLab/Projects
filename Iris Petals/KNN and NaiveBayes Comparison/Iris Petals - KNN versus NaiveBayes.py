
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



# Program --------------------------------------------------------------

print("\n ****  Iris Petals - KNN versus Naive Bayes  **** \n")


Filename = "iris_test.csv"
DF = pd.read_csv(Filename, sep= ",")
DF_SafeCopy = DF.copy()


Target = "species"
Seed = 38


# Adjusting Target Info (Cat. Text to Cat. Numeric)

Species_Dict = { "setosa": 0,
                 "versicolor": 1,
                 "virginica": 2 }

DF[Target] = DF[Target].map(Species_Dict)


# KNN Solution (Best n= 5*) --------------------------------------------

n = 5

imputer = KNNImputer(n_neighbors= n,
                     weights= "uniform", metric= "nan_euclidean")

DF_Transform = imputer.fit_transform(DF)
DF_Transform = pd.DataFrame(data= DF_Transform, columns= DF.columns)

DF_Transform[Target] = DF_Transform[Target].apply(lambda x: np.round(x, decimals= 0))



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

Y_Pred = Categorical_NaiveBayes.predict(X_Test)
Acc_Test = metrics.accuracy_score(Y_Test, Y_Pred)*100

print(f" Categorical Naive Bayes Accuracy = {Acc_Test:.2f}%")


# Applying Categorical Naive Bayes

Missing_X = DF_Missing.drop(columns= [Target])
Missing_Y = Categorical_NaiveBayes.predict(Missing_X)

DF_Missing[Target] = Missing_Y

DF_NB = DF_Ready.append(DF_Missing).sort_index(ascending= True)


# **** Comparing Real Set, KNN and Categorical Naive Bayes ****

DF_Compare = pd.read_csv("iris.csv", sep= ",")

Species_Dict = { 0: "setosa",
                 1: "versicolor",
                 2: "virginica" }

# Adding KNN Column

Data = DF_Transform["species"].map(Species_Dict)
DF_Compare["species_KNN"] = Data


# Adding CatNB Column

Data = DF_NB["species"].map(Species_Dict)
DF_Compare["species_NB"] = Data


# Saving .csv for futher Analysis

DF_Compare.to_csv("Iris_KNN_and_NB.csv", sep= ",", index= False)


# Accuracy Measuring

True_Info = DF_Compare["species"]
KNN_Info = DF_Compare["species_KNN"]
NB_Info = DF_Compare["species_NB"]

Conf_Matrix_Labels = ["setosa", "versicolor", "virginica"]

Conf_Matrix_KNN = confusion_matrix(True_Info, KNN_Info, labels= Conf_Matrix_Labels)
print(Conf_Matrix_KNN)
print("")

Conf_Matrix_NB = confusion_matrix(True_Info, NB_Info, labels= Conf_Matrix_Labels)
print(Conf_Matrix_NB)
print("")


# Closing

print("\n * \n")

