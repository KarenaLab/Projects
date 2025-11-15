# [P336] Iris Dataset - Decision tree model

# Insights, improvements and bugfix
#


# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from graphviz import Source

# Project libraries
from src.iris_tools import (load_dataset, target_split, organize_report)


# Functions -------------------------------------------------------------
def clf_decisiontree(x_train, x_test, y_train,
                     criterion="gini", max_depth=None, max_leaf_nodes=None,
                     min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0,
                     random_state=None, graphviz=True):
    
    clf = DecisionTreeClassifier()

    # Hyperparams
    clf.criterion = criterion
    clf.max_depth = max_depth
    clf.max_leaf_nodes = max_leaf_nodes
    clf.min_samples_leaf = min_samples_leaf
    clf.min_samples_split = min_samples_split
    clf.min_weight_fraction_leaf = min_weight_fraction_leaf
    clf.random_state = random_state

    # Fit and predict
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Parameters
    params = dict()

    # GraphViz
    if(graphviz == True):
        pass

        #dot_data = export_graphviz(clf)
        #f = open("dot_data.dot", mode="w", encoding="utf-8")
        #f.write(dot_data)
        #f.close()

        #src = Source.from_file(filename="dot_data.dot")
        #src.render(filename="dot_data", format="png", view=True)
        
        #os.remove("dot_data.dot")
        # https://www.pythontutorials.net/blog/converting-dot-to-png-in-python/


    return y_pred, params


# Setup/Config ----------------------------------------------------------



# Program ---------------------------------------------------------------
df = load_dataset()
target = "species"

# Data Split
x, y = target_split(df, target=target)
x = x[["petal_length_cm", "petal_width_cm"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=162)

y_pred, _ = clf_decisiontree(x_train, x_test, y_train,
                             max_depth=2)



# Scout theme: "Always leave the campsite cleaner than you found it"
organize_report() 

# end

