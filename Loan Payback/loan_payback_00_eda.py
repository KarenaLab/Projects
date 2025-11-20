# [P531] Loan payback - S5_E11
# EDA

# Insights, improvements and bugfix
#


# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt


# Personal modules
from src.loan_payback_tools import (load_dataset,
                                    cols_categoric, cols_numeric)

from src.plot_histbox import plot_histbox


# Functions



# Setup/Config
savefig = True


# Program --------------------------------------------------------------
df = load_dataset(filename="train.csv", path=".\database")
target = "loan_paid_back"

sample = df.sample(n=10000, random_state=314)
for col in cols_numeric():
    plot_histbox(data=sample[col], title=f"Loan payback S5E11 - Histogram {col}",
                 savefig=savefig)


# end
