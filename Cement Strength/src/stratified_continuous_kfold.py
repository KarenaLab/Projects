# [P305] Stratified Continuous KFold
# Applies the stratified KFold for targets wirh Continuous


# Insights, improvements and bugfix
# 01 - Allow the user to inform the binning strategy name
# 02 - Move ´bin_strategy´ out for a callable function
#


# Libraries
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold


# Setup/Config
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
def stratified_continuous_kfold(DataFrame, target, n_splits=5, bins=None,
                                random_state=None, shuffle=True):
    """
    Performs Stratified KFold with continuous target.
    Returns a list with **n_splits** tuples with:
    (folder no, train_index, test_index)

    """
    # Data preparation
    data = DataFrame.copy()

    # Uses **target** continous to split into bins (Eureca)
    x, y = _target_split(data, target=target)

    # Define the number of minimum bins
    bins_list = list()
    no_bins = np.inf
    bins_strategy = ["sturges", "fd", "doane", "scott", "stone", "rice", "sqrt"]

    for bs in bins_strategy:
        _, bs_edges = np.histogram(y, bins=bs)

        if(bs_edges.size < no_bins):
            no_bins = bs_edges.size
            bins_edges = bs_edges[:]

      
    # Creating segments from the continuous target
    segment_list = list()

    for value in y:
        segment = np.sum(value > bins_edges)

        if(segment == 0):
            # Solving the problem of creating bins with number 0 and
            # colapsing it with the first one bin.
            # Technically will be only one value that will do it.
            segment = 1

        segment_list.append(segment)


    data["bin"] = segment_list


    # Stratified KFold
    # Uses the bin column as the target as categories
    x, y = _target_split(data, target="bin")

    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    skf.get_n_splits(x, y)

    kf_indexes = list()
    for i, (train_index, test_index) in enumerate(skf.split(x, y)):
        fold = (i, train_index, test_index)
        kf_indexes.append(fold)


    return kf_indexes


def _target_split(DataFrame, target):
    """
    Splits variable(s) and **target** from the **DataFrame**

    """   
    x = DataFrame.drop(columns=[target])
    y = DataFrame[target]

    return x, y

    
# end
