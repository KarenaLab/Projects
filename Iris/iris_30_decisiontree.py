# [P336] Iris Dataset

# Insights, improvements and bugfix
#


# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


# Personal libraries
from src.iris_tools import (load_dataset, target_split, scaler,
                            organize_report)


# Functions ------------------------------------------------------------



# Setup/Config ---------------------------------------------------------



# Program --------------------------------------------------------------
df = load_dataset()
target = "species"

x, y = target_split(df, target=target)



