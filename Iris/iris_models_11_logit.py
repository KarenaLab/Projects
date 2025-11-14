# Iris - Project [P336]

# Libraries
import os
import sys
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.stats as st

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from iris_tools import *

# Personal modules
sys.path.append(r"c:\python_modules")


# Functions



# Setup/Config



# Program --------------------------------------------------------------
df = load_dataset()
x, y = target_split(df, target="species")

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7,
                                                    random_state=137)

x_train, x_test = scaler(x_train, x_test, method="Standard")


# end

