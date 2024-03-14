# AutoMPG [P316]


# Insights, improvements and bugfix
#

# Libraries
import numpy as np
import pandas as pd
import scipy.stats as st

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt



# ----------------------------------------------------------------------
def load_dataset():
    """
    Load and prepare **AutoMPG** dataset for further analysis and
    modeling.

    """
    filename = "auto_mpg.csv"
    data = pd.read_csv(filename, sep=",", encoding="utf-8")
    data = data_preparation(data)

    return data


def data_preparation(DataFrame, clean_dataframe=True):
    """
    Prepares AutoMPG dataframe.

    """
    data = DataFrame.copy()

    # Columns names
    col_names = dict()

    for old_name in data.columns:
        new_name = old_name.lower()
        new_name = new_name.replace(" ", "_")\
                           .replace("-", "_")

        col_names[old_name] = new_name

    data = data.rename(columns=col_names)

    # Units
    data["kpl"] = data["mpg"] * 0.42514371
    data = data.drop(columns=["mpg"])

    data["displacement_cm3"] = data["displacement"] * 16.387064
    data = data.drop(columns=["displacement"])

    data = data.rename(columns={"horsepower": "power_hp"})

    data["weight_kg"] = data["weight"] * 0.45359237
    data = data.drop(columns=["weight"])

    # Numeric
    data["power_hp"] = pd.to_numeric(data["power_hp"], errors="coerce")

    # Clean DataFrame
    if(clean_dataframe == True):
        data = data.dropna()
        data = data.drop_duplicates()
        data = data.reset_index(drop=True)

    
    return data


def target_split(DataFrame, target):
    """
    Splits a dataframe into variables and target.

    """
    x = DataFrame.drop(columns=[target])
    y = DataFrame[target]

    return x, y


def kfold_generate(DataFrame, n_splits=5, random_state=None, shuffle=True):
    """
    Creates KFolds.
    Returns the **indexes** for train and test. Works with fold_split.

    """
    # Fold strategy
    kf = KFold()

    # Folding hyperparameters
    kf.n_splits = n_splits
    kf.random_state = random_state
    kf.shuffle = shuffle

    # Folds
    kf_indexes = list()
    for i, (train_index, test_index) in enumerate(kf.split(DataFrame)):
        fold = (i, train_index, test_index)
        kf_indexes.append(fold)


    return kf_indexes


def fold_split(DataFrame, train_index, test_index, target):
    """
    Prepares a train and test dataframes for a given fold.
    Works with kfold_generate.

    """
    # Variables columns
    x_cols = DataFrame.columns.to_list()
    x_cols.remove(target)

    # Data split
    x_train = DataFrame.loc[train_index, x_cols]
    x_test = DataFrame.loc[test_index, x_cols]
    y_train = DataFrame.loc[train_index, target]
    y_test = DataFrame.loc[test_index, target]


    return x_train, x_test, y_train, y_test
    

def scaler(x_train, x_test, method=StandardScaler()):
    """
    Applies the **method** (that should be a scikit-learn function)
    to x_train and x_test.

    """
    sc = method

    # Fit and transform
    sc = sc.fit(x_train)

    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)

    return x_train, x_test


def powertransform(x_train, x_test, method="yeo-johnson", standardize=True):
    """
    Applies the **method** (Yeo-Johnson or Box-Cox) to x_train and
    x_test.

    """
    pt = PowerTranformer()

    # Hyperparameters
    pt.method = method
    pt.standardize = standardize

    # Fit and tranform
    pt = pt.fit(x_train)

    x_train = pt.transform(x_train)
    x_test = pt.transform(x_test)

    return x_train, x_test


def regr_linregr(x_train, x_test, y_train, y_test, fit_intercept=True,
                 positive=False):
    """
    Applies a **Linear Regression** model

    """
    # Model
    regr = LinearRegression()

    # Hyperparameters
    regr.fit_intercept = fit_intercept
    regr.positive = positive

    # Fit and predict
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    parameters = {"intercept":regr.intercept_, "coefs":regr.coef_}

    # Metrics
    results = regr_metrics(y_test, y_pred)


    return parameters, results


def regr_lasso(x_train, x_test, y_train, y_test, fit_intercept=True,
                 positive=False, alpha=1, max_iter=1000, tol=0.0001):
    """
    Applies a **Lasso** linear regression model.

    """
    # Model
    regr = Lasso()

    # Hyperparameters
    regr.fit_intercept = fit_intercept
    regr.positive = positive
    regr.alpha =alpha
    regr.max_iter = max_iter
    regr.tol = tol

    # Fit and predict
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    parameters = {"intercept":regr.intercept_, "coefs":regr.coef_}

    # Metrics
    results = regr_metrics(y_test, y_pred)


    return parameters, results


def regr_ridge(x_train, x_test, y_train, y_test, fit_intercept=True,
                 positive=False, alpha=1, max_iter=None, tol=0.0001):
    """
    Applies a **Ridge** linear regression model.

    """
    # Model
    regr = Ridge()

    # Hyperparameters
    regr.fit_intercept = fit_intercept
    regr.positive = positive
    regr.alpha =alpha
    regr.max_iter = max_iter
    regr.tol = tol

    # Fit and predict
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    parameters = {"intercept":regr.intercept_, "coefs":regr.coef_}

    # Metrics
    results = regr_metrics(y_test, y_pred)


    return parameters, results
    

def regr_metrics(y_true, y_pred):
    """
    Calculates the metrics regression.
    
    """
    # Calc
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pearson = st.pearsonr(y_true, y_pred).statistic

    results = {"mae":mae, "rmse":rmse, "r2":r2, "pearson":pearson}
    

    return results


def store_results(storage, new_line):
    new_line = pd.Series(data=new_line)

    storage = pd.concat([storage, new_line.to_frame().T], ignore_index=True)
    
    return storage


def find_optimum(metric_list, param_list, best="min"):
    """
    Finds the **best** value (min or max) in the **metric_list** and
    returns the value in **param_list** and **metric_list**.

    """
    # Define the best function
    if(best == "min" or best == "minimum"):
        func = np.min

    elif(best == "max" or best == "maximum"):
        func = np.max

    # Finder
    best_index = metric_list.index(func(metric_list))

    best_param = param_list[best_index]
    best_metric = metric_list[best_index]

    results = {"param": best_param, "value": best_metric}

    return results

def model_report(data, confidence=0.95, decimals=6):
    """
    Creates an easy way to look for the model performance metrics
    confidence.

    """
    data = np.array(array)

    mean = np.round(np.mean(data), decimals=decimals)
    stddev = np.round(np.std(data), decimals=decimals)
    ic = np.round(confidence_interval(data, confidence=confidence), decimals=decimals)

    results = {"mean": mean, "stddev": stddev, "ic": ic}

    return results


def confidence_interval(data, confidence=0.95):
    """


    """
    data = np.array(data)

    ic = st.t.interval(confidence=confidence,
                       df=(data.size - 1),      # Degrees of Freedom
                       loc=np.mean(data),
                       scale=st.sem(data))      # Standard Error of Mean

    return ic

