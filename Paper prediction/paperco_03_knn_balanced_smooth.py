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
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

import matplotlib.pyplot as plt


# Project libraries
from src.paper_co_tools import (load_dataset, remove_cols_unique, feat_eng_runtime_inv,
                                add_failure_tag, remove_df_columns, organize_report,
                                cols_tags)

  

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


    return data


def remove_cols_for_train(DataFrame, columns):
    """
    Remove the given **columns** from **DataFrame**.
    Important: That function is out of `prepare_dataset` due
    data balacing strategies.

    Arguments:
    * DataFrame: Pandas DataFrame
    * columns: List with columns names to be removed*

    Output:
    * DataFrame: Pandas dataframe processed.

    """
    # Columns preparation
    cols_remove = list()
    cols_dataframe = list(DataFrame.columns)

    # Check column names match
    for col in columns:
        if(cols_dataframe.count(col) == 1):
            cols_remove.append(col)


    DataFrame = DataFrame.drop(columns=cols_remove)

    return DataFrame
    

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
    * https://vitalflux.com/hold-out-method-for-training-machine-learning-model
    
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
    data_trainval = data_trainval.reset_index(drop=True)
    
    data_test = DataFrame.loc[test, :]
    data_test = data_test.reset_index(drop=True)
    

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


def balance_near(DataFrame, threshold=-40):
    """
    *** Function that will work ONLY with Paper CO Project ***

    Function will balance the *DataFrame*, keeping the values **closer/near**
    the threshold of the machine failure of -20.

    The objective is to keep 50% (or close) of dataset above -20 runtime threshold
    and other 50% (or close) under it. For this reason the limit is drawn to [-40, 0].

    Arguments:
    * DataFrame: Paper CO project dataframe,
    * threshold: Limit of time to be considered to balance the dataframe
                 (default=-40),

    Output:
    * DataFrame: Pandas dataframe processed.

    """
    # Rule for NEAR balanced DataFrame.
    DataFrame = DataFrame[DataFrame["runtime_inv"] >= threshold]
    DataFrame = DataFrame.reset_index(drop=True)

    return DataFrame


def balance_uniform(DataFrame, random_state=42):
    """
    *** Function that will work ONLY with Paper CO Project ***

    Function will balance the *DataFrame keeping uniformly balanced values through
    all period of time of non_failure. The objective is to keep 50% of data uniformly
    distributed in non-failure range of time and other 50% of data between -20 and 0,
    the failure_flag range.

    Arguments:
    * DataFrame: Paper CO project dataframe,
    * random_state: Seed for repeatibility (default=42),

    Output:
    * DataFrame: Pandas dataframe processed.

    """
    data = pd.DataFrame(data=[])

    for i in DataFrame["asset_id"].unique():
        info = DataFrame.groupby("asset_id").get_group(i)

        run_ok = info[info["failure_flag"] == 0]
        run_ok = run_ok.sample(n=20, replace=False, random_state=random_state)
        data = pd.concat([data, run_ok])

        run_not_ok = info[info["failure_flag"] == 1]
        data = pd.concat([data, run_not_ok])


    data = data.reset_index(drop=True)

    return data


def balance_far(DataFrame):
    """
    *** Function that will work ONLY with Paper CO Project ***

    Function will balance the *DataFrame*, keeping the values **far/away**
    the threshold of the machine failure of -20.

    The objective is to keep 50% (or close) of dataset of first moments of return of
    equipment, so, far from the problem  and other 50% (or close) under it. For this
    reason the limit is drawn for two moments: Not failure the first 20 moments from
    machine start and Failure the last 20 moments from the break-down.

    Arguments:
    * DataFrame: Paper CO project dataframe,

    Output:
    * DataFrame: Pandas dataframe processed.

    """
    data = pd.DataFrame(data=[])

    for i in DataFrame["asset_id"].unique():
        info = DataFrame.groupby("asset_id").get_group(i)

        run_ok = info[info["runtime"] <= 20]
        data = pd.concat([data, run_ok])

        run_not_ok = info[info["failure_flag"] == 1]
        data = pd.concat([data, run_not_ok])


    data = data.reset_index(drop=True)

    return data
        
    
def apply_scaler(x_train, x_test, scaler=StandardScaler()):
    """
    Apply scikit-learn scaler method to the train dataset and,
    replicate the same parameters of train dataset to test dataset.

    Arguments:
    * x_train: Pandas **train** dataframe,
    * x_test: Pandas **test** dataframe,
    * scaler: Scaling method (Imported only StandardScaler() and
              MinMaxScaler(), (default=StandardScaler())

    Output:
    * x_train: Scaled **train** Pandas dataframe,
    * x_test: Scaled **test** Pandas dataframe,
    
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



def apply_pca(x_train, x_test=None, n_components=None, output="pandas"):
    """
    Apply scikit-learn PCA method to to the **train** dataset and, IF GIVEN,
    replicate the same parameters of train dataset to test dataset.

    Arguments:
    * x_train: Pandas **train** dataframe,
    * x_test: Pandas **test** dataframe,
    * n_components: Number of components to use in PCA. If does not informed (None),
                    will use the full size of original dataset (default=None),
    * output: Pandas dataframe or NumPy array (default=Pandas)

    Output:
    * x_train: Transformed **train** Pandas dataframe,
    * x_test: Transformed **test** Pandas dataframe or None,
    * results: Dictionary with compnents, explained_variance_ratio and noise_variance.

    More about noise_variance: [1] PRML (Bishop), p.574,
    [2] http://www.miketipping.com/papers/met-mppca.pdf
    
    """
    
    # Components preparation
    if(n_components == None):
        n_components = x_train.shape[1]

    # PCA
    pca = PCA()

    # Hyperparams
    pca.n_components = n_components

    # Fit and transform
    pca.fit(x_train)

    x_train = pca.transform(x_train)
    if(type(x_test) != "NoneType"):
        x_test = pca.transform(x_test)

    # Output (Pandas DataFrame or NumPy array)
    if(output == "pandas"):
        col_names = [f"PC_{i+1}" for i in range(0, x_train.shape[1])]
        x_train = pd.DataFrame(data=x_train, columns=col_names)

        if(type(x_test) != "NoneType"):
            x_test = pd.DataFrame(data=x_test, columns=col_names)


    # Results
    results = dict()
    results["components"] = pca.components_
    results["explained_variance_ratio_"] = pca.explained_variance_ratio_
    results["noise_variance_"] = pca.noise_variance_

       
    return x_train, x_test, results

def apply_sma(DataFrame, columns, window):
    """


    """
    # Columns preparation
    cols_sma = list()
    cols_dataframe = list(DataFrame.columns)

    for i in columns:
        if(cols_dataframe.count(i) == 1):
            cols_sma.append(i)


    # SMA - Simple Moving Average Smooth
    for col in cols_sma:
        DataFrame[col] = DataFrame[col].rolling(window=window).mean()

    DataFrame = DataFrame.dropna()
    DataFrame = DataFrame.reset_index(drop=True)

    return DataFrame
    
            
def clf_kneighbors(x_train, x_test, y_train, n_neighbors=2, weights="uniform"):
    """
    Apply K_Neigbors Classifier model to dataset (x_train, x_test and y_train),
    considering 02 (two) main hyperparameters: n_neighbors and weights,

    Arguments:
    * x_train: Variables to be used for **train**,
    * x_test: Variables to be used for **test**,
    * y_train: Target to be used for **train**,
    * n_neighbors: Number of neighbors for decision,
    * weights: Weight to use for error. Could be uniform or distance. (default=uniform)

    """
    # KNeighbors Classifier
    clf = KNeighborsClassifier()

    # Hyperparams
    clf.n_neighbors = n_neighbors
    clf.weights = weights

    # Fit and predict
    clf.fit(x_train, y_train)
    y_pred_test = clf.predict(x_test)
    y_pred_train = clf.predict(x_train)
    

    # Parameters
    params = dict()
    

    return y_pred_test, y_pred_train, params


def clf_metrics(y_true, y_pred, show_matrix=False):
    """
    Performs **Confusion Matrix** from given model based in comparing
    **y_true** and **y_pred**. Optionally, could print in terminal the
    Confusion Matrix.

    Returns primary and some secondary metrics based in Confusion Matrix.


    Arguments:
    * y_true: NumPy array or python list with real values (truth/true),
    * y_pred: NumPy array or python list with predicted values (from model),
    * show_matrix: True or False*. Show in terminal the matrix.

    Output:
    * results: dictionary with primary and some secondary metrics based in
               Confusion Matrix (TP, FN, FP and TN; TPR (or Recall), FPR, FNR,
               TNR (or Specificity), Accuracy, Precision and F1 Score).

    """
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Results
    results = dict()

    #                      Predicted
    #                 |  True  |  False
    #         --------|--------|---------
    #            True |   TP   |   FN 
    # Actual  --------|--------|---------
    #           False |   FP   |   TN
    #         ---------------------------

    # Primary metrics    
    results["tn"] = tn
    results["fp"] = fp
    results["fn"] = fn
    results["tp"] = tp

    # Secondary metrics
    results["tpr"] = tp / (tp + fn)                         # True Positive Rate = Recall, Detection Rate, Sensitivity)                         
    results["fpr"] = fp / (fp + tn)                         # False Positive Rate = Type I Error (Incorrectly predicts positive)
    results["fnr"] = fn / (fp + tp)                         # False Negative Rate = Type II Error (Incorrectly predicts negative)
    results["tnr"] = tn / (fp + tn)                         # True Negative Rate = Specificity    
    results["acc"] = (tp + tn) / (tp + tn + fp + fn)        # Accuracy    
    results["prec"] = tp / (tp + fp)                        # Precision    
    results["f1_score"] = (2 * tp) / ((2 * tp) + fp + fn)   # F1 Score = Harmonic median between Precision [prec] and Recall [tpr]

    if(show_matrix == True):
        print_confusion_matrix(results)
        

    return results         


def print_confusion_matrix(results):
    """
    Based in results from a confusion matrix given as a **dictionary**.
    Print the Matrix in Terminal.

    Arguments:
    * results: Dictionary with (at least) with four primary values of matrix,

    Output:
    None (Print in terminal)

    """
    # Data to string
    tn = str(results["tn"])
    fp = str(results["fp"])
    fn = str(results["fn"])
    tp = str(results["tp"])

    # Matrix print
    print(f">       |       True |      False |")
    print(f"  ---------------------------------")
    print(f"   True | {tp:>5s} [TP] | {fn:>5s} [FN] |")
    print(f"  False | {fp:>5s} [FP] | {tn:>5s} [TN] |\n")

    return None



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
df, df_test = holdout_split(df, target, test_size=.3, random_state=314)
df = apply_sma(df, columns=cols_tags(), window=6)
df = balance_uniform(df)

folds = kfold_split(df, target, n_splits=n_splits, random_state=314)

df_results = pd.DataFrame(data=[])
for n in range(3, 20+1):
    print(f"> n_neigbors: {n}")

    # Cross-Validation    
    for i, [train_index, test_index] in folds.items():       
        # 1- Variables and Target split        
        x, y = target_split(df, target=target)
        cols_remove = ["asset_id", "runtime", "runtime_inv","setting_1", "setting_2"]
        x = remove_cols_for_train(x, columns=cols_remove)
        
        x_train, x_test = x.loc[train_index, :], x.loc[test_index, :]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        # 2- Scaler
        x_train, x_test = apply_scaler(x_train, x_test, scaler=StandardScaler())

        # 3- PCA
        #x_train, x_test, pca_results = apply_pca(x_train, x_test, n_components=2)
       
        # 4.1- Model: K Neighbors
        y_pred_test, y_pred_train, y_params = clf_kneighbors(x_train, x_test, y_train, n_neighbors=n, weights="uniform")
        train_results = clf_metrics(y_train, y_pred_train, show_matrix=False)
        test_results = clf_metrics(y_test, y_pred_test, show_matrix=True)

        # 4.2- Store results (further analysis)
        for metric, tag in zip([train_results, test_results], ["train", "test"]):
            for key, value in metric.items():
                df_results.loc[n, f"fold_{i}_{key}_{tag}"] = value

                
# Calculate Mean and Standard Deviation of Fold for main metric
for metric in ["fnr", "tnr", "tpr", "fpr"]:
    for i in df_results.index:
        values = np.array([])
        for j in range(0, n_splits):
            values = np.append(values, df_results.loc[i, f"fold_{j}_{metric}_test"])
            
        df_results.loc[i, f"{metric}_mean_test"] = np.mean(values)
        df_results.loc[i, f"{metric}_stddev_test"] = np.std(values)

        values = np.array([])
        for j in range(0, n_splits):
            values = np.append(values, df_results.loc[i, f"fold_{j}_{metric}_train"])

        df_results.loc[i, f"{metric}_mean_train"] = np.mean(values)
        df_results.loc[i, f"{metric}_stddev_train"] = np.std(values)


print("")
print(df_results[["fnr_mean_train", "fnr_mean_test", "fnr_stddev_test"]])


# Scout theme: "Always leave the campsite cleaner than you found it"
organize_report(src=path_main, dst="report")

