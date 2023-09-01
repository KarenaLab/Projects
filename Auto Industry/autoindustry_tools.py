
# Libraries
import numpy as np
import pandas as pd



def read_csv(filename, sep=",", encoding="utf-8"):
    """
__pycache__   

    """
    data = pd.read_csv(filename, sep=sep, encoding=encoding)


    return data


def dataframe_preparation(DataFrame, verbose=True):
    """
    Prepares dataframe reading.

    """
    data = DataFrame.copy()
    
    data = data.drop_duplicates()
    data = data.dropna()


    return data


def units_conversion(DataFrame, verbose=True):
    """
    

    """
    data = DataFrame.copy()

    data["consumption_kpl"] = data["mpg"].apply(lambda x: np.round(x * 0.42514371, decimals=3))
    data = data.drop(columns=["mpg"])

    data["displacement_cm3"] = data["displacement"].apply(lambda x: np.round(x * 16.387064, decimals=4))
    data = data.drop(columns=["displacement"])

    data["weight_kg"] = data["weight"].apply(lambda x: np.round(x * 0.45359237, decimals=1))
    data = data.drop(columns=["weight"])

    data = data.rename(columns={"horsepower": "power_hp"})
    data = data.rename(columns={"acceleration": "acceleration_s"})
    
                     
    return data

