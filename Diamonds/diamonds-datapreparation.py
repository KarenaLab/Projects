
# Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Setup/Config
seed = 27


# Functions



# Program --------------------------------------------------------------
filename = "diamonds.csv"
df = pd.read_csv(filename, sep=",", encoding="utf-8")

for col in df.columns:
    unique = df[col].nunique()
    


