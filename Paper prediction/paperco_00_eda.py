# Project Paper CO

# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt


# Project libraries
from src.paper_co_tools import load_dataset


# Functions



# Setup/Config
path_main = os.getcwd()
path_database = os.path.join(path_main, "database")
path_report = os.path.join(path_main, "report")


# Program ---------------------------------------------------------------
df = load_dataset(filename="pm_train.txt", path=path_database)
