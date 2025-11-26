# [P529] Penguins project

# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt


# Personal modules
from src.penguins_tools import load_csv


# Functions
def cols_numeric():
    cols = ["culmen_length_mm", "culmen_depth_mm",
            "flipper_length_mm", "body_mass_g"]

    return cols


def cols_categoric():
    cols = ["island", "gender"]

    return cols


# Setup/Config



# Program --------------------------------------------------------------
df = load_csv()
target = "species"



# end
