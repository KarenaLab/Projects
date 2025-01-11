# P472 - GEFCom2014 Project (Time Series)

# Libraries
import os
import sys

import numpy as np
import pandas as pd


# Setup/Config
path_main = os.getcwd()
path_data = r"D:\01 - Projects Binder\472 - GEFCom2014\data"

# Functions
def load_dataset(filename, path=None):
    # Filename preparation
    if(path == None):
        path = os.getcwd()

    filename = os.path.join(path, filename)

    # Load data
    data = pd.read_csv(filename, index_col=0, sep=",", encoding="utf-8")

    return data



# Program --------------------------------------------------------------
filename = "GEFCom2014-E.csv"
df = load_dataset(filename)

    

    
    
