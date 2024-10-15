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



# Setup/Config
path_main = os.getcwd()



# Program --------------------------------------------------------------
filename = "iris_sql.db"

files = list(os.listdir())
if(files.count(filename) == 1):
    os.remove(filename)
    print(f' > File "{filename}" already exists, file deleted')


conn = sqlite3.connect(filename)
cursor = conn.cursor()

db_schema = """ CREATE TABLE iris
                (sepal_length FLOAT,
                 sepal_width FLOAT,
                 petal_length FLOAT,
                 petal_width FLOAT,
                 species TEXT)
            """

cursor.execute(db_schema)
print(f' > New file "{filename}" created (empty database)')

cursor.close()
conn.close()

# end
