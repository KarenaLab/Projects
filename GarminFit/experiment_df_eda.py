
# Libraries
import os

import numpy as np
import pandas as pd

import datetime as dt
from zoneinfo import ZoneInfo


# Project libraries
from source.utils_df_ops import *



# Functions
def fit_dataprep(DataFrame, timezone=None):
    # Timezone adjust
    if(timezone != None):
        DataFrame["timestamp"] = DataFrame["timestamp"].apply(lambda x: x.astimezone(ZoneInfo(timezone)))

    DataFrame = DataFrame.set_index(keys="timestamp", drop=True)

    # Rename columns
    cols_rename = {"position_lat": "latitude",
                  "position_long": "longitude",
                  "distance": "distance_m",
                  "enhanced_speed": "speed_mps",
                  "enhanced_altitude": "altitude_m",
                  "power": "power_w"}

    DataFrame = DataFrame.rename(columns=cols_rename)

    # Data adjust: Latitude and Longitude
    # Data is stored in semicircles


    # New variable: Pace (Speed in km per min)
    DataFrame["pace_km_min"] = DataFrame["speed_mps"].apply(lambda x: calc_speed_to_pace(x, decimals=4))
                  

    return DataFrame


def calc_speed_to_pace(speed, decimals=None):
    """
    Speed must be in meters per second (m/s)

    """
    # Tranformation (m/s to km/min)
    if(speed > 0):
        speed = (speed * (60/1000))
        pace = 1 / speed

    else:
        pace = 0


    if(isinstance(decimals, int) == True):
        pace = np.round(pace, decimals=decimals)


    return pace


def calc_semicircle_to_deg(value):
    """
    Trasform value in semicircle to degree.
                            180
    Eq: deg = semicircle x ------
                            2^31
    """
    deg = (value * 180) / 2**31

    return deg
 
def count_nan(DataFrame):
    for col in DataFrame.columns:
        nans = pd.isnull(DataFrame[col]).sum()
        print(col, nans)


    return None


def check_unknown(DataFrame):
    for col in DataFrame.columns:
        if(col.count("unknown") != -1):
            print(col)


    return None



# Program ---------------------------------------------------------------
df = import_fit(filename="23000350509_ACTIVITY.fit", path=r"./activities")
df = fit_dataprep(df, timezone="America/Sao_Paulo")

