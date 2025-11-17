# [P529] Penguins project

# Insights, improvements and bugfix
#


# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
def load_csv():
    """
    Load Penguins project dataset.

    """
    path = r"D:\01 - Projects Binder\529 - Penguins Project\database"
    filename = "penguins.csv"

    data = pd.read_csv(os.path.join(path, filename),
                       sep=",", encoding="utf-8")

    return data    



