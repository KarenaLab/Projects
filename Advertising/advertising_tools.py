# Advertising (ISLP) [P425]

# Versions
# 01 - May 10th, 2024 - Starter
# 02 -


# Insights, improvements and bugfix
#


# Libraries
import numpy as np
import pandas as pd
import scipy.stats as st

import matplotlib.pyplot as plt



# ----------------------------------------------------------------------
def load_dataset_advertising():
    filename = "advertising.csv"
    data = pd.read_csv(filename, sep=",", encoding="utf-8")

    col_names = dict()
    for old_name in data.columns:
        new_name = old_name.lower()
        col_names[old_name] = new_name

    data = data.rename(columns=col_names)
    

    return data
