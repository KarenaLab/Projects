
# Libraries
import os
import warnings
import logging

import datetime as dt
import numpy as np
import pandas as pd
import scipy.stats as stats

import streamlit as st

import matplotlib.pyplot as plt

# Project libraries
from source.utils_garminfit import (files_list, files_filter, file_unzip,
                                    get_datetime)

from source.utils_df_ops import (import_fit)
                                    


# Setup: Version control
VERSION = "1.01"

# Setup: Streamlit HTML page
st.set_page_config(page_title=f"GarminFit Explorer - version {VERSION}",
                   page_icon="images\S_letter_blue.png",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Setup: Warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Config
path_main = os.getcwd()


# Program ---------------------------------------------------------------
NOW = get_datetime(seconds=True)

# Import user config
app_settings = dict()


# Sidebar ---------------------------------------------------------------
with st.sidebar:
    pass
    # Expander logo
    # Mode selection



