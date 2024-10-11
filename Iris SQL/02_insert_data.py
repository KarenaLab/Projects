# Name [P36] - Iris SQL
# Iris as SQL database for basic/intermediate functions learning.


# Libraries
import os
import sys
from collections import namedtuple

import sqlite3

import numpy as np
import pandas as pd


# Personal modules
sys.path.append(r"c:\python_modules")


# Functions
def read_csv(filename, sep=",", encoding="utf-8"):
    data = pd.read_csv(filename, sep=sep, encoding=encoding)

    return data


def database_connection(filename):
    global conn, cursor

    conn = sqlite3.connect(filename)
    cursor = conn.cursor()

    return None


def database_close():
    global conn, cursor

    cursor.close()
    conn.close()

    return None


def insert_row(sl, sw, pl, pw, s, verbose=False):
    row_header = "INSERT INTO iris (sepal_length, sepal_width, petal_length, petal_width, species)"
    row_values = f"VALUES ({sl}, {sw}, {pl}, {pw}, '{s}');"

    row = row_header + " " + row_values

    cursor.execute(row)
    conn.commit()

    if(verbose == True):
        print(row)

    return None


# Setup/Config
path_main = os.getcwd()



# Program --------------------------------------------------------------
database_connection("iris_sql.db")

df = read_csv("iris.csv")
for row in df.index:
    sl = df.loc[row, "sepal_length"]
    sw = df.loc[row, "sepal_width"]
    pl = df.loc[row, "petal_length"]
    pw = df.loc[row, "petal_width"]
    s = df.loc[row, "species"]

    insert_row(sl, sw, pl, pw, s, verbose=True)


database_close()

# end
