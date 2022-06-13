
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



# Definitions ----------------------------------------------------------

Seed = 27
np.random.seed(Seed)

plt.style.use("ekc")



# Program --------------------------------------------------------------

print("\n ****  Kaggle - Feature Engineering - Autos  **** \n")

DF = pd.read_csv("autos.csv", sep= ",")


# Feature Engineering

DF["stroke_ratio"] = DF["stroke"]/DF["bore"]
DF["displacement"] = np.pi * ((0.5*DF["bore"])**2) * DF["stroke"] * DF["num_of_cylinders"]


# Closing

print("\n * \n")

