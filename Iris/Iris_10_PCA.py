
# Libraries
import numpy as np
import pandas as pd


# Personal libraries
import sys
sys.path.append(r"C:\python_modules")

from data_preparation import *
from pca_analysis import *


# Setup/Config


# Program --------------------------------------------------------------
filename = "iris.csv"
df = pd.read_csv(filename, sep=",", encoding="utf-8")

x, y = split_target(df, target="species")

df_pca, pca_variance = pca_analysis(x, title=f"Iris - PCA Analysis", yellowline=95, greenline=99,
                                    savefig=False)
