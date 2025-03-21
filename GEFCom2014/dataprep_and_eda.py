# P472 - GEFCom2014 Project (Time Series)

# Libraries
import os
import sys
import warnings

import datetime as dt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Setup/Config



# Functions
def load_dataset():
    """
    Load the complete **dataset** from GEFCom2014 project.
    Focused in Time Series learning methods.

    """
    # Load data
    filename = "GEFCom2014-E.csv"
    data = pd.read_csv(filename, index_col=0, sep=",", encoding="utf-8")

    # Prepare columns names
    col_names = dict()
    for old_name in data.columns:
        new_name = old_name.lower()
        col_names[old_name] = new_name

    data = data.rename(columns=col_names)
    data = data.rename(columns={"t": "temp_c",
                                "load": "load_kwh"})

    # Prepare date and time
    data["hour"] = data["hour"].apply(_hour_to_string)  
    data["datetime"] = data["date"] + " " + data["hour"]
    data["datetime"] = pd.to_datetime(data["datetime"], format="%Y-%m-%d %H:%M")
    data.index = data["datetime"]
    data = data.drop(columns=["date", "hour", "datetime"])

    # Prepare data format (temperature)
    data["temp_c"] = data["temp_c"].apply(lambda x: coerce_nan(x, float, fahrenheit_to_celsius))

    # Prepare index frequency
    data = data.asfreq(freq="H")

        
    return data


def data_preparation(DataFrame, start_time=None, end_time=None):
    """
    Slices the DataFrame between **start_time** and **end_time**.
    Important: Inclusive in both sides: [start_time, end_time].

    """
    # Slice start_time
    if(isinstance(start_time, str) == True):
        start_time = pd.to_datetime(start_time, format="%Y-%m-%d")
        DataFrame = DataFrame[DataFrame.index >= start_time]

    if(isinstance(end_time, str) == True):
        end_time = pd.to_datetime(end_time, format="%Y-%m-%d")
        DataFrame = DataFrame[DataFrame.index <= end_time]

        
    return DataFrame


def _hour_to_string(hour):
    """
    Internal function.
    Transforms an hour informed as an integer(1; 24) and returns the
    hour as a string pre-formated to be used as datetime stamp (HH:MM)
    and (00:00; 23:59)

    """
    # Substitute hour 24 per 0
    if(hour == 24):
        hour = 0

    # Insert a zero suffix for two digits padded hour
    if(hour <= 9):
        hour_suffix = "0"
    else:
        hour_suffix = ""

    # Create %HH:%MM string
    string = f"{hour_suffix}{hour}:00"

    return string


def check_nans(DataFrame, columns="all", decimals=2, verbose=True):
    """
    Check the number of NaNs (Not a Number) in given **columns**.
    Columns could be a list if desired some specific columns to check.

    """
    # Columns preparation
    if(columns == "all" or columns == None):
        columns = list(DataFrame.columns)

    # NaNs check
    nans_register = dict()

    no_rows = DataFrame.shape[0]
    for col in columns:
        no_nans = DataFrame[col].isna().sum()
        pct_nans = np.round((no_nans / no_rows) * 100, decimals=decimals)

        nans_register[col] = [no_nans, pct_nans]
        print(f" > {col}: NaNs= {no_nans} ({pct_nans}%)")

    # [1] Create an function to print it more friendly
    # [2] Create a plot to receive this information and transform it
    # into a plot
    
    return nans_register


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE) stands for the percentual
    absolute error of the prediction (or forecast).
    [MLTSF p.47]
    
    More info:
    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    https://www.statology.org/how-to-interpret-mape/

    """
    # Data preparation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculation
    result = np.mean((np.abs(y_true - y_pred) / y_true))

    return result


def ts_decomposition(DataFrame, model="additive", filt=None, period=None):
    """
    Performs the Time Series decomposition and returns a friendly
    output to be vizualized, maybe it is the first view of the Time
    Series data.

    Variables:
    * model: additive (default) or multiplicative.

    More info:
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.DecomposeResult.html
    
    """
    # Model check
    model = model.lower()
    if(model != "additive" and model != "multiplicative"):
        model = "additive"
        warnings.warn('Model Error: Selected model "additive" as default.')

    # Time Series decomposition
    decomposition = sm.tsa.seasonal_decompose(DataFrame, model=model, filt=filt, period=period)

    observed = decomposition.observed
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    weights = decomposition.weights

    # Return decomposition as a dataframe
    data = pd.DataFrame(data=[])
    for info in [observed, trend, seasonal, residual]:
        data[info.name] = info

    # Append `weights` only if values are different from scalar (1).
    if(len(weights.unique()) > 1):
        data[weights.name] = weights
        

    return data


def create_lagged_features(DataFrame, variable, max_lag, freq):
    """
    Lagged features are create with the assumption that what happened
    in the past can influence or contain a sort of intrinsic information
    about the future.

    Function will create [1, **max_lag**] variables (columns) with
    **freq** based in the given **variable**.

    Variables
    * DataFrame: Pandas dataframe where data is,
    * variable: Single variable (as a string) to be shifted,
    * max_lag: integer or a list. If given an integer, function will create
               a list from [1, max_lag] and interate the shift with this.
               Also could inform directly the desired list. Inform a list
               with a single value if you wish a single new variable creation.
    * freq: Timestamp frequency to create lagged variables.

    More info:
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html
    https://medium.com/@rahulholla1/advanced-feature-engineering-for-time-series-data-5f00e3a8ad29
    
    """
    # `max_lag` preparation:
    if(isinstance(max_lag, int) == True):
        lag_list = range(1, max_lag+1)

    elif(isinstance(max_lag, list) == True):
        lag_list = max_lag[:]

    # Lagged variables
    for t in lag_list:
        new_variable = f"{variable}_lag{t}"
        DataFrame[new_variable] = DataFrame[variable].shift(t, freq=freq)


    return DataFrame

def create_rolling_window_stats(DataFrame, variable, window, stats=["min", "mean", "max"],
                                decimals=None):
    """
    Compute statistics on the values from a given **variable** by defining
    a range called *window* that means this number of samples before the
    sample used.

    Variables
    * DataFrame: Pandas dataframe where data is,
    * variable: Single variable (as a string) to be windowed,
    * window: integer number to consider as a past window,
    * stats: Statistics to calculate the rolling window values.
             Default is: ["min, "mean", "max"]

    List of stats available:


    More info:
    https://pandas.pydata.org/pandas-docs/stable/reference/window.html
    https://pandas.pydata.org/pandas-docs/stable/user_guide/window.html
    
    """
    # Window treatment
    info = DataFrame[variable]
    info = info.shift(window-1)
    info = info.rolling(window=window)

    # Create Windowed response
    data = pd.DataFrame(data=[])
    for s in stats:
        if(s == "min"):
            data[f"min_w{window}"] = info.mean()

        elif(s == "mean"):
            data[f"mean_w{window}"] = info.min()

        elif(s == "max"):
            data[f"max_w{window}"] = info.max()


    # Apply rounded values (if called):
    if(isinstance(decimals, int) == True):
        for col in data.columns:
            data[col] = np.round(data[col], decimals=decimals)

    # Append the final target
    data[variable] = DataFrame[variable]

    return data


def create_expanded_window_stats(DataFrame, variable, stats=["min", "mean", "max"], decimals=None):
    """
    Compute statistics that include all previous data.

    """
    # Window treatment
    info = DataFrame[variable]
    info = info.expanding()

    # Create Windowed response
    data = pd.DataFrame(data=[])
    for s in stats:
        if(s == "min"):
            data["min_expand"] = info.mean()

        elif(s == "mean"):
            data["mean_expand"] = info.min()

        elif(s == "max"):
            data["max_expand"] = info.max()


    # Apply rounded values (if called):
    if(isinstance(decimals, int) == True):
        for col in data.columns:
            data[col] = np.round(data[col], decimals=decimals)

    # Append the final target
    data[variable] = DataFrame[variable]

    return data           


def fahrenheit_to_celsius(temp_f, decimals=3):
    """
    Tranforms the temperature in Fahrenheit (°F) to Celsius (°C).
    Equation: Temp_C = (Temp_F - 32) * (9/5)

    """
    temp_c = (temp_f - 32) * (9 / 5)
    temp_c = np.round(temp_c, decimals=decimals)

    return temp_c


def celsius_to_fahrenheit(temp_c, decimals=3):
    """
    Transforms the temperature in Celsius (°C) to Fahrenheit (°F).
    Equation: Temp_F = ((9/5) * Temp_C) + 32

    """
    temp_f = ((9/5) * temp_c) + 32
    temp_f = np.round(temp_f, decimals=decimals)

    return temp_f


def coerce_nan(value, kind, function):
    """
    Function to help lambdas functions to handle missing data or data
    not standartized.

    Variables:
    * value: Value to be checked, transformed by function if follows the
             kind,
    * kind: Type of data that function expects, any data out of this
            will be coerced to np.nan,
    * function: Function to be applied to the value if its follows the
                expected kind.
    

    """
    if(isinstance(value, kind) == True and pd.isnull(value) == False):
        answer = function(value)

    else:
        answer = np.nan

    return answer


def scaler_standard(Series):
    """
    Performs a Standard normalization, rescaling data for values with
    mean in zero and standard deviatin of 1.

    Returns:
    values = Pandas series with standard scaler applied.
    params = Python dictionary with value to invert the scaling.
             params[col_name] = [method, mean, stddev] 

    """
    # Standard scaling
    x_mean = Series.mean()
    x_stddev = Series.std()

    params = dict()
    params[Series.name] = ["standard", x_mean, x_stddev]

    def standard(x, mean, stddev):
        x_scaled = (x - mean) / stddev
        return x_scaled

    Series = Series.apply(lambda x: standard(x, x_mean, x_stddev))

    return Series, params 
        

def inv_scaler_minmax(Series, params):
    """
    Performs the inverse of Min-Max normalization, rescaling the data
    for the previous (real) data.

    Returns:
    values = Pandas series with inverse min-max applied.

    """   
    _, x_min, x_max = list(params.values())[0]

    def inv_minmax(x_scaled, minimum, maximum):
        x = ((maximum - minimum) * x_scaled) + minimum
        return x

    Series = Series.apply(lambda xs: inv_minmax(xs, x_min, x_max))

    return Series


def inv_scaler_standard(Series, params):
    """
    Performs the inverse of Standard normalization, recaling the data
    for the previous (real) data.

    Return values = Pandas series with inverse standard applied.

    """
    _, x_mean, x_stddev = list(params.values())[0]

    def inv_standard(x_scaled, mean, stddev):
        x = (x_scaled * stddev) + mean
        return x

    Series = Series.apply(lambda xs: inv_standard(xs, x_mean, x_stddev))

    return Series

    
# Program --------------------------------------------------------------
df = load_dataset()
df = data_preparation(df, start_time="2012-01-01", end_time="2014-12-31")

decomposition = ts_decomposition(df["load_kwh"])
df_window = create_rolling_window_stats(df, variable="load_kwh", window=4,
                                        decimals=0)

df_expand = create_expanded_window_stats(df, variable="load_kwh")

# end
