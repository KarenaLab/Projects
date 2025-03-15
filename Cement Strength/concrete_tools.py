# [P504] Concrete compressive strength


# Libraries
import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt



# ----------------------------------------------------------------------
def load_dataset():
    filename = "concrete_strength.csv"
    data = pd.read_csv(filename, sep=",", encoding="utf-8")

    return data

