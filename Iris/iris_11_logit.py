# Iris - Project [P336]

# Libraries
import os
import sys
from collections import namedtuple

import numpy as np
import pandas as pd

import scipy.stats as st

import matplotlib.pyplot as plt

from iris_tools import *

# Personal modules
sys.path.append(r"c:\python_modules")


# Functions



# Setup/Config



# Program --------------------------------------------------------------
df = load_dataset()
x, y = target_split(df, target="species")


# end

