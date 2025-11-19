# [P531] Loan payback - S5_E11

# Insights, improvements and bugfix
#


# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
def load_dataset(filename, path=None):
    # Path preparation
    if(path != None):
        filename = os.path.join(path, filename)

    # Load .csv
    data = pd.read_csv(filename, index_col=0, sep=",", encoding="utf-8")


    return data


