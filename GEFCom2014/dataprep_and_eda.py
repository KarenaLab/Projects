# P472 - GEFCom2014 Project (Time Series)

# Libraries
import os
import sys

import datetime as dt
import numpy as np
import pandas as pd

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
    data = data.rename(columns={"t": "temperature"})

    # Prepare date and time
    data["hour"] = data["hour"].apply(_hour_to_string)  
    data["datetime"] = data["date"] + " " + data["hour"]
    data["datetime"] = pd.to_datetime(data["datetime"], format="%Y-%m-%d %H:%M")
    data.index = data["datetime"]
    data = data.drop(columns=["date", "hour", "datetime"])

    # Prepare data format (temperature)
    data["temperature"] = data["temperature"].apply(lambda x: coerce_nan(x, float, fahrenheit_to_celsius))

    # Prepare index frequency
    data = data.asfreq(freq="H")
    
    
    return data


def data_preparation(DataFrame, start_time, end_time):
    """
    Slices the DataFrame between **start_time** and **end_time**.
    Important: Inclusive in both sides: [start_time, end_time].

    """
    # Date and Time prep
    for var in [start_time, end_time]:
        var = pd.to_datetime(var, format="%Y-%m-%d")

    # DataFrame slicing [start_time, end_time]
    DataFrame = DataFrame[DataFrame.index >= start_time]
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
    if(model != "additive" or model != "multiplicative"):
        model = "additive"
        print(f' > Warning: Selected model "additive" as default.')

    # Time Series decomposition
    decomposition = sm.tsa.seasonal_decompose(DataFrame, model=model, filt=filt, period=period)

    observed = decomposition.observed
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    weights = decomposition.weights


    return None


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
df = data_preparation(df, "2012-01-01", "2014-12-31")

    
# end
