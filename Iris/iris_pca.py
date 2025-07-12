# [Pxxx] Project name
# (optional) Short description of the program/module.


# Libraries
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 

import matplotlib.pyplot as plt


# Personal modules
from iris_tools import (load_dataset)


# Functions
def apply_pca(DataFrame, n_components=None):
    """
    Performs PCA Analysis for a given **DataFrame**, this data HAVE TO
    BE without target, only variables and it is optional to give the
    **n_components**.

    Argument:
    * DataFrame: Pandas dataframe with variables only,
    * n_components: Number of components to apply (optional), if not
                    informed, will use the size of the original data,
    

    """
    # Data Preparation
    scaler = StandardScaler()
    scaler.fit(DataFrame)
    x_scaled = scaler.transform(DataFrame)
    _, n_cols = x_scaled.shape

    # PCA: Transformation
    pca = PCA(n_components=n_cols)
    pca.fit(x_scaled)
    x_pca = pca.transform(x_scaled)

    # PCA: Output as a DataFrame [1]
    labels = [f"PC{i}" for i in range(1, n_cols+1)]
    x_pca = pd.DataFrame(data=x_pca, columns=labels)

    # PCA: Statistics [2]
    results = dict()
    results["explained_variance"] = pca.explained_variance_ratio_
    results["cumulative_explain"] = np.cumsum(pca.explained_variance_ratio_)
    

    return x_pca, results 
    
    
# Setup/Config



# Program --------------------------------------------------------------
df = load_dataset()
target = "species"

df_pca = df.drop(columns=[target])
df_pca, results = apply_pca(df_pca)


# end
