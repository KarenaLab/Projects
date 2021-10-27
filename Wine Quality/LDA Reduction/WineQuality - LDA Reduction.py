
# Wine Quality Red

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt


print("\n ****  Wine Quality Red - LDA - Linear Discriminant Analysis  **** \n")


DF = pd.read_csv("winequality_red.csv", sep= ",")


# Dataset Adjusts

Quality_Map = { 3: 0,
                4: 1,
                5: 2,
                6: 3,
                7: 4,
                8: 5 }

DF["quality"] = DF["quality"].map(Quality_Map)


# Target Setting

Target = "quality"

Data_X = DF.drop(columns = [Target])
Data_Y = DF[Target]


# Defining Max number of Components
# Eq = min(No of Features and (Target_Classes - 1))

Data_Features = Data_X.shape[1]
Target_Classes = Data_Y.nunique()

Comp_Max = min(Data_Features, (Target_Classes - 1))


VarianceRatio_List = []
Components_List = range(1, Comp_Max+1)

for n in Components_List:

    LDA = LinearDiscriminantAnalysis(n_components= n)
    LDA_Model = LDA.fit(Data_X, Data_Y)

    LDA_X = LDA_Model.transform(Data_X)

    VarianceRatio = LDA.explained_variance_ratio_
    VarianceRatio = np.sum(VarianceRatio)

    VarianceRatio_List.append(VarianceRatio)
    
    print(f" > Explained Variance Ratio with {n} components: {VarianceRatio:.5f}")


print("")


# Plotting

plt.style.use("ekc")

fig = plt.figure()

Title = "Wine Quality Red - LDA Reduction"
plt.suptitle(Title, fontsize= 14)

plt.plot(Components_List, VarianceRatio_List, color= "navy")
plt.axhline(y= 0.95, color= "darkred", linestyle= "--", linewidth= 0.5)
plt.axhline(y= 0.99, color= "orange", linestyle= "--", linewidth= 0.5)

plt.xlabel("No of Components")
plt.ylabel("Explained")

plt.xticks(Components_List)

#plt.savefig(Title, dpi= 240)
plt.show()


# Closing

print(" * \n") 



