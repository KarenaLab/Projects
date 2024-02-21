# Iris - Project [P336]
# Learning machine learning fundamentals with Iris project

# Libraries
import numpy as np
import pandas as pd
import scipy.stats as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt



# ----------------------------------------------------------------------
def load_dataset():
    """
    Extract, transform and load data from Iris dataset and prepares it
    for further analysis and processing.

    """
    # Data load
    filename = "iris.csv"
    data = pd.read_csv(filename, sep=",", encoding="utf-8")

    # Columns preparation
    col_name = dict()
    
    for old_name in data.columns:
        new_name = old_name.replace("length", "length_cm")\
                           .replace("width", "width_cm")

        col_name[old_name] = new_name


    data = data.rename(columns=col_name)
            

    return data


def target_split(DataFrame, target):
    """
    Splits **DataFrame** into **x** and **y**, 

    """
    x = DataFrame.drop(columns=[target])
    y = DataFrame[target]

    return x, y


def scaler(x_train, x_test, method="Standard"):
    """
    Applies **Scaler** to variables.
    Could be **StandardScaler** or **MinMaxScaler**

    """
    if(method == "Standard" or method == "MinMax"):
        # Select method
        if(method == "Standard"):
            sc = StandardScaler()

        elif(method == "MinMax"):
            sc = MinMaxScaler()

        # Fit and transform
        sc = sc.fit(x_train)
        x_train = sc.transform(x_train)
        x_test = sc.transform(x_test)
        
    else:
        # Method is not valid
        print(f" >>> Warning: Invalid method, no scaler applied")


    return x_train, x_test


# end
