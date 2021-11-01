
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



# Program ---------------------------------------------------------------

print("\n ****  Iris Petals - KNN Imputation  **** \n")

Filename = "iris_test.csv"
DF = pd.read_csv(Filename, sep= ",")


DF_Train = DF.dropna().reset_index(drop= True)

Target = "species"
Remove = 25
Seed = 38


DF_Train_Rows = DF_Train.shape[0]
DF_Train_Cols = DF_Train.shape[1]
DF_Train_Columns = DF_Train.columns

np.random.seed(Seed)

Remove = int((Remove/100)*DF_Train_Rows)
Remove_List = np.random.randint(low=0, high= DF_Train_Rows, size= Remove)

for i in Remove_List:

    DF_Train.loc[i, Target] = np.nan
   

print(f" > Removing {Remove} items from {Target} to Test MICE. \n")


# Converting Text Feature to Numeric Values

Species_Dict = { "setosa": 0,
                 "versicolor": 1,
                 "virginica": 2 }

Data_True = DF_Train[Target].map(Species_Dict)

DF_Train[Target] = Data_True

Total = Data_True.value_counts().sum()

Ref_0 = Data_True.value_counts()[0]/Total
Ref_1 = Data_True.value_counts()[1]/Total
Ref_2 = Data_True.value_counts()[2]/Total


# MICE (Multivariate Imputation by Chained Equation)


N_Min = 5
N_Max = 15

N_Neighbours_List = [0]
Dif_0_List = [Ref_0]
Dif_1_List = [Ref_1]
Dif_2_List = [Ref_2]


for n in range(N_Min, (N_Max+1)):

    print(f" > Calculating with {n} MICE Max Iter")

    Imputer = IterativeImputer(imputation_order= "ascending", max_iter= n,
                               n_nearest_features= 5, random_state= 42)

    DF_Transform = Imputer.fit_transform(DF_Train)
    DF_Transform = pd.DataFrame(data= DF_Transform, columns= DF_Train_Columns)


    # Adjusting Information

    Data_Pred = DF_Transform["species"]
    Data_Pred = Data_Pred.apply(lambda x: np.round(x, decimals= 0))
    
    Train_Total = Data_Pred.value_counts().sum()
    
    Train_0 = Data_Pred.value_counts()[0]/Train_Total
    Train_1 = Data_Pred.value_counts()[1]/Train_Total
    Train_2 = Data_Pred.value_counts()[2]/Train_Total

    N_Neighbours_List.append(n)
    Dif_0_List.append(Train_0)
    Dif_1_List.append(Train_1)
    Dif_2_List.append(Train_2)


# Plotting

plt.style.use("ekc")

fig = plt.figure()

Title = "Iris Petals - MICE Preparation - Best Iter"
plt.suptitle(Title, fontsize= 14)

plt.plot(N_Neighbours_List, Dif_0_List, color= "navy", label= "setosa")
plt.plot(N_Neighbours_List, Dif_1_List, color= "darkred", label= "versisolor")
plt.plot(N_Neighbours_List, Dif_2_List, color= "orange", label= "virginica")

plt.axhline(y= Ref_0, color= "navy", linestyle= "--", linewidth= 0.5)
plt.axhline(y= Ref_1, color= "darkred", linestyle= "--", linewidth= 0.5)
plt.axhline(y= Ref_2, color= "orange", linestyle= "--", linewidth= 0.5)

plt.axvline(x= 10, color= "black", linestyle= "--", linewidth= 0.5)

plt.xlabel("No of Max Iter")
plt.ylabel("% of Sample")

plt.legend()

plt.savefig(Title, dpi= 240)
plt.show()
   

# Closing

print("\n * \n")
