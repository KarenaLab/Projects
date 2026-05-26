
# Libraries
import os

import datetime as dt
import numpy as np
import pandas as pd
import scipy.stats as stats

import fitdecode

import matplotlib.pyplot as plt


# Setup/Config



# Functions
def import_fit(filename, path=None):
    """


    """
    # Path preparation
    if(path != None):
        filename = os.path.join(path, filename)


    # .fit reading
    data = list()

    with fitdecode.FitReader(filename) as fit:
        for frame in fit:         
            if(frame.frame_type == fitdecode.FIT_FRAME_DATA):
                if(frame.name == "record"):
                    row = {field.name: field.value for field in frame.fields}
                    data.append(row)

    # DataFrame
    data = pd.DataFrame(data=data)
                  
    return data


