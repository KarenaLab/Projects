
import numpy as np
import pandas as pd


# Program ---------------------------------------------------------------

print("\n ****  Iris Petals - Preparing NaNs  **** \n")

Filename = "iris.csv"
DF = pd.read_csv(Filename, sep= ",")

Target = "species"
Remove = 35
Seed = 27


DF_Rows = DF.shape[0]
np.random.seed(Seed)

Remove = int((Remove/100)*DF_Rows)
Remove_List = np.random.randint(low= 0, high= DF_Rows, size= Remove)

for i in Remove_List:

    DF.loc[i, Target] = np.nan


print(f" > Removing {Remove} items from {Target} column.")

Filename = "iris_test.csv"
DF.to_csv(Filename, sep= ",", index= False)

print(f" > File {Filename} for KNN and GaussNB Test saved.")


# Closing

print("\n * \n")
