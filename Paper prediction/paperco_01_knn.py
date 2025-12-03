# Project Paper CO

# Libraries
import os
import itertools
import warnings

from fractions import Fraction

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt


# Project libraries
from src.paper_co_tools import (load_dataset, remove_cols_unique, feat_eng_runtime_inv,
                                add_failure_tag, remove_df_columns, organize_report)

from src.plot_histbox import plot_histbox
from src.plot_barv import plot_barv
from src.plot_scatterhist import plot_scatterhist
from src.plot_heatmap import plot_heatmap



# Functions -------------------------------------------------------------
def prepare_dataset(filename, path):
    """
    Import and prepare **DataFrame** for modeling.

    Arguments:
    * filename: string with filename and extension,
    * path: (Optional) If file is not in root path,

    Return:
    * data: Pandas Dataframe
    
    """
    data = load_dataset(filename=filename, path=path)
    data = remove_cols_unique(data, verbose=False)
    data = feat_eng_runtime_inv(data)
    data = add_failure_tag(data, threshold=-20)

    cols_remove = ["asset_id", "runtime", "tag_6", "runtime_inv","setting_1", "setting_2"]
    data = remove_df_columns(data, columns=cols_remove)

    return data


def target_split(DataFrame, target):
    """
    Splits **DataFrame** into x (variables) and y(**target**).

    Arguments:
    * DataFrame: Pandas DataFrame with project data,
    * target: Target for x (variables) and y (target) split,

    Returns:
    x: DataFrame with all variables,
    y: DataFrame (Series) with target,

    """
    x = DataFrame.drop(columns=[target])
    y = DataFrame[target]

    return x, y
    

def holdout_split(DataFrame, target, test_size=0.2, random_state=42):
    """
    Splits DataFrame into **two** Dataframes for Train, Validation and Test.

    Arguments:
    * DataFrame: Pandas DataFrame with project data,
    * target: Target for Stratified KFold keep the proportions constant,
    * test_size: Size of test dataframe,
    * random_state: Seed for tests reproducibility,

    Returns:
    * trainval: Pandas DataFrame to be used for Train and Validation
    * test: Pandas DataFrame to be used for Test.

    More info about the holdout strategy:
    

    """
    # test_size preparation
    test_size = np.round(test_size, decimals=2)
    test_size = Fraction(test_size).limit_denominator()

    splits_get = test_size.numerator
    splits_total = test_size.denominator  

    # Stratified KFold
    skf = StratifiedKFold()

    # Hyperparams
    skf.n_splits = splits_total
    skf.shuffle = True
    skf.random_state = random_state

    # Target split
    x, y = target_split(DataFrame, target=target)

    # trainval and test split
    trainval = np.array(DataFrame.index)
    test = np.array([])
    
    for i, (_, test_index) in enumerate(skf.split(x, y)):
        test = np.append(test, test_index)

        # test array is an array to be removed from trainval array.
        items_to_keep = ~np.isin(trainval, test)
        trainval = trainval[items_to_keep]

        # Will only keep the "numerator" size for test
        if((i + 1) >= splits_get):
            break


    # DataFrame split
    data_trainval = DataFrame.loc[trainval, :]
    data_test = DataFrame.loc[test, :]

    return data_trainval, data_test


def kfold_split(DataFrame, target, n_splits=5, random_state=42):
    """
    Splits DataFrame into **n_splits** splits for train and test.
    Important: Keeps the proportion of target in each split.

    Arguments:
    * DataFrame:
    * target: Target for Stratified KFold keep the proportions constant,
    * n_splits: [int] Number of splits for train and test,
    * random_state: Seed for tests reproducibility,

    Returns:
    * splits: [dict] Dictionary with split number, train indexes and test indexes,

    More info about the Strat strategy:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

    """
    # Stratified KFold
    skf = StratifiedKFold()

    # Hyperparams
    skf.n_splits = n_splits
    skf.shuffle = True
    skf.random_state = random_state

    # Target split
    x, y = target_split(DataFrame, target=target)

    # Folds
    folds = dict()
    for i, (train_index, test_index) in enumerate(skf.split(x, y)):
        folds[i] = [train_index, test_index]


    return folds


def data_scaler(x_train, x_test, scaler=StandardScaler()):
    """


    """
    # Initialize scaler method
    # Important: Only imported StandardScaler and MinMaxScaler
    scaler = scaler

    col_names = x_train.columns
    # Fit and transform
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)
    x_train = pd.DataFrame(data=x_train, columns=col_names)

    x_test = scaler.transform(x_test)
    x_test = pd.DataFrame(data=x_test, columns=col_names)
    

    return x_train, x_test


def clf_kneighbors(x_train, x_test, y_train, n_neighbors=2, weights="uniform"):
    """


    """
    # KNeighbors Classifier
    clf = KNeighborsClassifier()

    # Hyperparams
    clf.n_neighbors = n_neighbors
    clf.weights = weights

    # Fit and predict
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Parameters
    params = dict()
    

    return y_pred, params


def clf_metrics(y_true, y_pred):
    """


    """
    # ROC (Receiver Operating Chracteristic Curve) and
    # AUC (Area Under Curve)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)

    # Results
    results = Results
    results.fpr = fpr
    results.tpr = tpr
    results.thresholds = thresholds
    results.auc_score = auc_score

    return results         
        


# Classes and Objects ---------------------------------------------------
class Results:
    def __init__(self, fpr, tpr, thresholds, auc_score):
        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thresholds
        self.auc_score = auc_score



# Setup/Config ----------------------------------------------------------
path_main = os.getcwd()
path_database = os.path.join(path_main, "database")
path_report = os.path.join(path_main, "report")

warnings.filterwarnings("ignore")

SAVEFIG = True



# Program ---------------------------------------------------------------

# Import dataset for Models
df = prepare_dataset(filename="pm_train.txt", path=path_database)
target = "failure_flag"

n_splits = 5
df_trainval, df_test = holdout_split(df, target, test_size=.2, random_state=314)
folds = kfold_split(df_trainval, target, n_splits=n_splits, random_state=314)

df_results = pd.DataFrame(data=[])
for n in range(3, 9+1):
    print(f" > n_neigbors: {n}")

    # Cross-Validation    
    for i, [train_index, test_index] in folds.items():
        x, y = target_split(df, target=target)

        x_train = x.loc[train_index, :]
        x_test = x.loc[test_index, :]
        y_train = y.loc[train_index]
        y_test = y.loc[test_index]

        # Pipeline
        # Normalization
        x_train, x_test = data_scaler(x_train, x_test, scaler=StandardScaler())

        # PCA
        

        
        # PCA
        # Balancing
        # k_neighbors

        """    
        y_pred, params = clf_kneighbors(x_train, x_test, y_train, n_neighbors=n)
        fold_results = clf_metrics(y_test, y_pred)

        df_results.loc[n, f"fold_{i}"] = fold_results.auc_score

    # Calculate Mean and Standard Deviation of Fold
    values = np.array([])
    for i in range(0, n_splits):
        values = np.append(values, df_results.loc[n, f"fold_{i}"])

    df_results.loc[n, "mean"] = np.mean(values)
    df_results.loc[n, "stddev"] = np.std(values)
    
"""    
  
# Scout theme: "Always leave the campsite cleaner than you found it"
organize_report(src=path_main, dst="report")

