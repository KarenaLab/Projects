
# Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow
# Aurélien Géron

# Cap.04 - Training Models - p.148~154


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

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

Y_Score = Logit.predict_proba(Data_X)[:, 1]

FPR, TPR, threshold = roc_curve(Data_Y, Y_Score)

Min_Distance = 1
x_min, y_min = 0, 0


i = 0
while(i < len(threshold)):

    x = FPR[i]
    y = TPR[i]

    Distance = np.sqrt((x-0)**2 + (1-y)**2)

    if(Distance < Min_Distance):

        Min_Distance = Distance

        x_min = x
        y_min = y
        threshold_min = threshold[i]


    i = i+1
        

# Plotting

plt.style.use("ekc")

fig = plt.figure()

Title = "Iris Petals - ROC Curve"
plt.suptitle(Title, fontsize= 14)

plt.plot(FPR, TPR, color= "orange", label= "ROC Curve")
plt.plot([0, 1], color= "navy", linestyle= "--", linewidth= 0.8, label= "Random Curve (0.5)")
plt.plot(x_min, y_min, marker = "o", color = "orange", )

plt.xlabel("FPR = False Positive Rate", loc= "center")
plt.ylabel("TPR = True Positive Rate", loc= "center")


plt.legend(loc= "lower right")
#plt.savefig(Title, dpi= 240)

plt.show()

# Closing

print(" * \n")

