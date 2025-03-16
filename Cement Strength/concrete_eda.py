# [P504] Cement compressive strength
# Performs EDA (Exploratory Data Analysis)

# Libraries
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt


# Personal modules
sys.path.append(r"c:\python_modules")
from eda_taxonomy import eda_taxonomy

from concrete_tools import load_dataset


# Functions



# Setup/Config



# Program --------------------------------------------------------------
df = load_dataset()
cols_class = eda_taxonomy(df, verbose=True)



# end
