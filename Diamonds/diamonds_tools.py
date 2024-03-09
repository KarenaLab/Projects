# Name [P063] - Diamonds
# Tools for the Diamond dataset project

# Versions
# 01 - Mar 07th, 2024 - Starter
# 02 -


# Insights, improvements and bugfix
#


# Libraries
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.stats as st

import matplotlib.pyplot as plt



# ----------------------------------------------------------------------
def load_dataset():
    """
    Loads and prepares the dataset for further analysis.
    
    """
    filename = "diamonds.csv"
    data = pd.read_csv(filename, sep=",", encoding="utf-8")
    
    return data    

