
# Libraries
import os
import shutil
import datetime as dt

from zipfile import ZipFile

import numpy as np
import pandas as pd
import scipy.stats as stats

import streamlit as st

import matplotlib.pyplot as plt


# Setup/Config



# Functions
def files_list(path=None):
    # Path preparation
    path_back = os.getcwd()

    if(path != None):
        os.chdir(path)

    # Files
    result = list()
    for f in os.listdir():
        if(os.path.isfile(f) == True):
            result.append(f)


    os.chdir(path_back)

    return result


def files_filter(array, extension):
    # Extension preparation
    extension = extension.replace(".", "")

    # Filter
    result = list()
    for f in array:
        name, ext = os.path.splitext(f)
        ext = ext.replace(".", "")

        if(extension == ext):
            result.append(f)


    return result


def file_unzip(filename, remove_zip=False, verbose=True):
    f = ZipFile(filename, mode="r")
    f.extractall(os.getcwd())
    f.close()

    if(verbose == True):
        print(f" > file extracted {filename}")

    if(remove_zip == True):
        os.remove(filename)

    return None


def unzip_activities():
    files = files_filter(files_list(), extension=".zip")

    for f in files:
        file_unzip(filename=f, remove_zip=True, verbose=True)

    return None


def store_activities(dst, verbose=True):
    files = files_filter(files_list(), extension=".fit")

    for f in files:
        source = os.path.join(os.getcwd(), f)
        destiny = os.path.join(dst, f)

        shutil.move(src=source, dst=destiny)

        if(verbose == True):
            print(f" > file moved {destiny}")

    return None


def get_datetime(seconds=True, microseconds=False):
    """
    Returns **date and time** from system with the format

    Arguments:
    * seconds: True or False (dafault=True). Selects if returns
               datetime with or without seconds. Important, if False
               will automatically also set microseconds as False
    * microseconds: True or False (default=False). Selects if returns 
                    datetime with or without microseconds,

    Output:
    * Datetime value as python datetime.
    
    """
    value = dt.datetime.now()

    if(seconds == False):
        value = value.replace(second=0)
        microseconds = False

    if(microseconds == False):
        value = value.replace(microsecond=0)
    
    return value  

