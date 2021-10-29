
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Program ---------------------------------------------------------------

print("\n ****  Missing Data Analysis  **** \n")

Filename = "hmeq.csv"
DF = pd.read_csv(Filename, sep= ",")

DF_Rows = DF.shape[0]
DF_Cols = DF.shape[1]

Counting_Dict = {}

for col in range(0, DF_Cols+1):

    Counting_Dict[col] = 0
    

for row in range(0, DF_Rows):

    Data = DF.iloc[row, :]
    count = Data.isna().sum()

    Counting_Dict[count] = Counting_Dict[count] + 1


Counting_Dict.pop(0)
Index = list(Counting_Dict.keys())
Values = list(Counting_Dict.values())


# Plotting

plt.style.use("ekc")

fig = plt.figure()

Title = "HMEQ - 01 - Missing Data Analysis"
plt.suptitle(Title, fontsize= 14)

plt.bar(Index, Values, color= "darkred", edgecolor= "black")

plt.xlabel("Number of missing data in each row", loc= "center")
plt.xticks(Index)

plt.ylabel("Count")
plt.grid(axis= "x")


#plt.savefig(Title, dpi= 240)
plt.show()


    

    

    




