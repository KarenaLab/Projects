# Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import KFold

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score



# Setup/Config
seed = 42



# Functions
def text_as_list(columns):
    """
    *** Internal function ***
    
    Verifies if columns are imputed as string, if it was, transforms it
    into a list.

    """

    if(isinstance(columns, str) == True):
        columns = columns.replace(",", "")
        columns = columns.strip()
        columns = columns.split()


    return columns


def nan_count(DataFrame, verbose=True):
    """
    Counts the number of NaNs (Not a Number) in each column of
    **DataFrame**. Returns a [0] dict with column name and the number
    of NaNs by column and [1] the total of the NaNs from Dataset.

    Verbose default is True, that prints columns information with NaNs
    and the total of NaNs from DataFrame.

    """

    nan_dict = {"name": "NaNs"}
    nan_total = 0

    nrows, ncols = DataFrame.shape
    
    for col in DataFrame.columns:
        nan_count = DataFrame[col].isna().sum()

        nan_dict[col] = nan_count
        nan_total = nan_total + nan_count

        if(nan_count > 0 and verbose == True):
            print(f" > Col {col} has {nan_count} NaNs ({nan_count/nrows*100:.3f}%)")


    if(verbose == True):
        print(f" > DataFrame has {nan_total} NaNs ({nan_total/(nrows*ncols)*100:.2f}%)")


    return nan_dict, nan_total


def repeat_count(DataFrame, verbose=True):
    """
    Counts the number of repeated rows in the **DataFrame** with
    2-Dimensions only and returns the [0] DataFrame without duplicates
    rows and the number of duplicated rows deleted.

    """

    if(DataFrame.ndim == 2):
        nrows_before = DataFrame.shape[0]
        DataFrame = DataFrame.drop_duplicates().reset_index(drop=True)
        nrows_after = DataFrame.shape[0]

        duplicates = nrows_before - nrows_after

        if(verbose == True):
            print(f" > DataFrame has {duplicates} duplicated rows ({(duplicates/nrows_before)*100:.3f}%)")

    else:
        duplicates = np.nan

        if(verbose == True):
            print(f" > Error: Dimension. DataFrame is not 2-dimensions")
        

    return DataFrame, duplicates


def remove_columns(DataFrame, columns, verbose=False):
    """
    Removes from **DataFrame** the **columns** listed.

    """

    columns = text_as_list(columns)
    DataFrame = DataFrame.drop(columns=columns)

    return DataFrame


def scaler_minmax(DataFrame, columns=None, param=None, verbose=False):
    """
    Applies the Min-Max transformation at given **DataFrame** and selected
    **columns**.

    If **param=None, Minimum and Maximum will be taken from column.
    If not, if param is a list [minimum, maximum], scaler will take this
       parameters.

    Returns the DataFrame modified and a dict with parameters used.

                 x - x_min
    Equation: ---------------
               x_max - x_min    (or x_range)
    
    """

    DF = DataFrame.copy()
    param_list = []
    
    # Column driven or Param driven
    if(columns != None):
        param = None
        columns = text_as_list(columns)
        _param = []

        for col in columns:
            data_min = DF[col].min()
            data_max = DF[col].max()
            _param.append(["minmax", data_min, data_max])            
     
    if(param != None):
        columns = []
        _param = []
        
        for i in range(0, len(param)):
            columns.append(param[i][0])
            _param.append(param[i][1])

    param = _param[:]
    
    for col, parameters in list(zip(columns, param)):
        strategy, data_min, data_max = parameters
                
        if(strategy == "minmax"):
            DF[col] = DF[col].apply(lambda x: ((x - data_min) / (data_max - data_min)))
            param_list.append([col, ["minmax", data_min, data_max]])

        if(verbose == True):
            data_min = np.round(data_min, decimals=6)
            data_max = np.round(data_max, decimals=6)
            print(f" > col {col} | min={data_min}, max={data_max}")
        

    return DF, param_list
    

def scaler_standard(DataFrame, columns=None, param=None, verbose=False):
    """
    Applies the Standard Score transformation at given **DataFrame** and
    selected **columns**.

    If **param**=None, Mean and Standard Deviation will be taken from column.
    If not, if param is a list [mean, stddev], scaler will take this parameters.

    Returns the DataFrame modified and a list with parameters used.

               x - x_mean
    Equation: ------------
                x_stddev
    
    """

    DF = DataFrame.copy()
    param_list = []

    # Column driven or Param driven
    if(columns != None):
        param = None
        columns = text_as_list(columns)
        _param = []

        for col in columns:
            data_mean = DF[col].mean()
            data_stddev = DF[col].std()
            _param.append(["standard", data_mean, data_stddev])

    if(param != None):
        columns = []
        _param = []

        for i in range(0, len(param)):
            columns.append(param[i][0])
            _param.append(param[i][1])

    param = _param[:]

    for col, parameters in list(zip(columns, param)):
        strategy, data_mean, data_stddev = parameters

        if(strategy == "standard"):
            DF[col] = DF[col].apply(lambda x: ((x - data_mean) / data_stddev))
            param_list.append([col, ["standard", data_mean, data_stddev]])

        if(verbose == True):
            data_mean = np.round(data_mean, decimals=6)
            data_stddev = np.round(data_stddev, decimals=6)
            print(f" > col {col} | mean={data_mean}, stddev={data_stddev}")


    return DF, param_list


def one_hot_encoding(DataFrame, column, prefix_sep="_"):
    """
    Performs One-Hot Encoding with the **column** in the **DataFrame**
    Returns the DataFrame with one-hot encoding columns.

    Using `pd.get_dummies` function. More [information]
    (https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)
    and `pd.merge` function. More [information]
    (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html)

    """

    ohe = pd.get_dummies(DataFrame[column], prefix=column, prefix_sep=prefix_sep)
    DataFrame = DataFrame.drop(columns=[column])
    DataFrame = pd.merge(left=DataFrame, right=ohe,
                         how="left", left_index=True, right_index=True)

    return DataFrame


def merge_dict(dict_left, dict_right):
    """
    Merges two dictionaries, called `dict_left` and `dict_right`.

    """

    inner = dict_left.copy()
    inner.update(dict_right)

    return inner


def invert_scaler(DataFrame, param, verbose=False):
    """
    Performs an inversion in scaler operation.
    Applies the **param** stored as a dict in the **Dataframe**.

    **param** is a dictionary with 03 (three) information:
       * [0]scaler applied: "minmax" or "standard"
       * if minmax: [1]minimum and [2]maximum of variable scaled,
       * if standard: [1]mean and [2]standard deviation of variable
         scaled.

       This information comes from scaler_minmax and scaler_standard.

    Returns the Dataframe with scaler inversion.
    
    """

    DF = DataFrame.copy()
    columns = list(param.keys())

    for col in columns:
        strategy, var_1, var_2 = param[col]
       
        if(strategy == "minmax"):
            data_min = var_1
            data_max = var_2

            if(verbose == True):
                print(f" > col {col} applying scaler inversion: min={data_min:.6f}, max={data_max:.6f}")

            DF[col] = DF[col].apply(lambda x: (x * (data_max - data_min)) + data_min)

            
        if(strategy == "standard"):
            data_mean = var_1
            data_stddev = var_2

            if(verbose == True):
                print(f" > col {col} applying scaler inversion: mean={data_mean:.6f}, stddev={data_stddev:.6f}")
            
            DF[col] = DF[col].apply(lambda x: ((x * data_stddev) + data_mean))


    return DF
  

def train_test_split(DataFrame, train_size=80, seed=None, verbose=False):
    """
    Splits **DataFrame** into train and test using **train_size**.
    **train_size** could be a number higher than 1, it will be read as
    int number 
    
    
    """
    if(train_size > 1):
        train_size = train_size / 100
            
    if(isinstance(seed, int) == True):
        np.random.seed(seed)

    nrows = DataFrame.shape[0]
    split = int(nrows * train_size)
    
    index = np.arange(start=0, stop=nrows, step=1, dtype=int)
    np.random.shuffle(index)

    train_index = index[ :split]
    test_index = index[split: ]

    DataFrame_train = DataFrame.loc[train_index, :].reset_index(drop=True)
    DataFrame_test = DataFrame.loc[test_index, :].reset_index(drop=True)

    if(verbose == True):
        print(f" > train_size: {int(train_size * 100)}% - Train rows: {DataFrame_train.shape[0]}, Test rows: {DataFrame_test.shape[0]}")


    return DataFrame_train, DataFrame_test


def target_split(DataFrame, target):
    """
    Splits **DataFrame** into two datasets, one (x_df) with
    independent variables and (y_df) with **target**.

    Target could be only one variable.

    """
    
    columns = DataFrame.columns.tolist()
    
    if(isinstance(target, str) == False):
        x_df, y_df = None, None
        print(f" > Error: target is not a string")
    

    elif(columns.count(target) == 0):
        x_df, y_df = None, None
        print(f" > Error: target not found at DataFrame")

    else:
        x_df = DataFrame.drop(columns=[target])
        y_df = DataFrame[target]
        

    return x_df, y_df


def pca(DataFrame, n_components=2):
    """


    """
    _pca = PCA(n_components=n_components)
    _pca.fit(DataFrame)
    explained_variance_ratio = _pca.explained_variance_ratio_
    


def model_linear_ols(x_train, y_train, x_test, fit_intercept=True, n_jobs=None, positive=False):
    """
    Aplies the **Linear Regression model** for the given **dataset**.
    Returns the predicted values and the parameters used at linear
    model.

    """
    # Setup Linear Regression
    model = LinearRegression()
    
    model.fit_intercept = fit_intercept
    model.n_jobs = n_jobs
    model.positive = positive   
    param = model.get_params()

    model.fit(x_train, y_train)
    model_intercept = model.intercept_
    model_coef = model.coef_
    y_pred = model.predict(x_test)

    return y_pred, param, model_intercept, model_coef


def model_linear_ridge(x_train, y_train, x_test, alpha=0.5, fit_intercept=True, max_iter=None, positive=False, random_state=None, solver="auto", tol=.0001):
    """
    Applies the **Linear Regression** with **Ridge** for the given
    **dataset**. Returns the predicted values and the parameters
    used at linear model.

    """
    # Setup Ridge
    model = Ridge()
    
    model.alpha = alpha
    model.fit_intercept = fit_intercept
    model.max_iter = max_iter
    model.positive = positive
    model.random_state = random_state
    model.solver = solver
    model.tol = tol
    param = model.get_params()

    model.fit(x_train, y_train)
    model_intercept = model.intercept_
    model_coef = model.coef_
    y_pred = model.predict(x_test)

    
    return y_pred, param, model_intercept, model_coef


def model_linear_lasso(x_train, y_train, x_test, alpha=0.5, fit_intercept=True, max_iter=1000, positive=False, precompute=False, random_state=None, selection="cyclic", tol=.0001, warm_start=False):
    """
    Applies the **Linear Regression** with **Lasso** for the given
    **dataset**. Returns the predicted values and the parameters
    used at linear model.

    """
    # Setup Lasso
    model = Lasso(alpha=alpha)
    
    model.alpha = alpha
    model.fit_intercept = fit_intercept
    model.max_iter = max_iter
    model.positive = positive
    model.precumpute = precompute
    model.random_state = random_state
    model.selection = selection
    model.tol = tol
    model.warm_start = warm_start
    param = model.get_params()

    model.fit(x_train, y_train)
    model_intercept = model.intercept_
    model_coef = model.coef_
    y_pred = model.predict(x_test)


    return y_pred, param, model_intercept, model_coef


def metrics_reg(y_true, y_pred, metrics="all", decimals=5, verbose=True):
    """
    Calculates some metrics for regression models between *y_true** and
    **y_pred**. The metrics used are controled by **metrics** variable
    and could be used the following items:
       * MAE = Mean Absolute Error
       * MSE = Mean Squared Error
       * MAPE = Mean Absolute Percentage Error
       * R2 = R Square
       * Pearson = Coeficient of Correlation (Pearson)
       * or *all* (default)

    
    """   
    metrics_dict = {}
    if(metrics.count("mae") == 1 or metrics.count("all") == 1):
        mae = mean_absolute_error(y_true, y_pred)
        mae = np.round(mae, decimals=decimals)

        metrics_dict["MAE"] = mae
        print(f" > MAE: {np.round(mae, decimals=decimals)}")       
        
    if(metrics.count("mse") == 1 or metrics.count("all") == 1):
        mse = mean_squared_error(y_true, y_pred)
        mse = np.round(mse, decimals=decimals)

        metrics_dict["MSE"] = mse
        print(f" > MSE: {np.round(mse, decimals=decimals)}")

    if(metrics.count("mape") == 1 or metrics.count("all") == 1):
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mape = np.round(mape, decimals=decimals)

        metrics_dict["MAPE"] = mape
        print(f" > MAPE: {np.round(mape, decimals=decimals)}")
    
    if(metrics.count("r2") == 1 or metrics.count("all") == 1):
        r2 = r2_score(y_true, y_pred)
        r2 = np.round(r2, decimals=decimals)

        metrics_dict["R2"] = r2
        print(f" > R2: {np.round(r2, decimals=decimals)}")

    if(metrics.count("pearson") == 1 or metrics.count("all") == 1):
        data = pd.DataFrame(data=[])
        data["y_true"] = y_true
        data["y_pred"] = y_pred
        pearson = data.corr(method="pearson")
        pearson = pearson.loc["y_true", "y_pred"]       

        metrics_dict["Pearson"] = pearson
        print(f" > Pearson: {np.round(pearson, decimals=decimals)}")

    if(verbose == True):
        print("")
        

    return metrics_dict

