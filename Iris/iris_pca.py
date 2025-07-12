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
from src.plot_pca_explain import plot_pca_explain



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


def plot_pca_principals(DataFrame, label, title=None, savefig=False, verbose=True):
    """


    """
    # Data preparation
    variables = DataFrame[label].unique()
    colors = ["navy", "darkred", "orange", "darkgreen", "darkviolet"]
    colors = colors[0: variables.size]

    # Title
    if(title == None):
        title = "PCA Principals"

    # RC Params
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["ps.papersize"] = "A4"
    plt.rcParams["xtick.direction"] = "inout"
    plt.rcParams["ytick.direction"] = "inout"
    plt.rcParams["xtick.major.size"] = 3.5
    plt.rcParams["ytick.major.size"] = 3.5

    # Plot
    fig = plt.figure(figsize=[6, 3.375])    # Widescreen 16:9
    fig.suptitle(title, fontsize=10, fontweight="bold", x=0.98, ha="right")

    for var, color in zip(variables, colors):
        data = DataFrame.groupby(by=label).get_group(var)
        plt.scatter(x=data["PC1"], y=data["PC2"], s=15, color=color, edgecolor="white",
                    alpha=0.6, label=var, zorder=20)

    plt.xlabel("PC1", loc="center")
    plt.ylabel("PC2", loc="center")

    plt.grid(axis="both", color="grey", linestyle="--", linewidth=0.5, zorder=5)       
    plt.legend(loc="best", framealpha=1).set_zorder(99)

    plt.tight_layout()

    # Printing 
    if(savefig == True):
        plt.savefig(title, dpi=320)

        if(verbose == True):
            print(f' > saved plot as "{title}.png"')

    else:
        plt.show()

    plt.close(fig)

    return None
    
    
# Setup/Config



# Program --------------------------------------------------------------
df = load_dataset()
target = "species"

df_pca = df.drop(columns=[target])
df_pca, results = apply_pca(df_pca)

#plot_pca_explain(results["explained_variance"], title="Iris - PCA Explained variance", savefig=False)

df_pca = df_pca[["PC1", "PC2"]]
df_pca[target] = df[target]

plot_pca_principals(df_pca, label=target, title="Iris - PCA Principals", savefig=True)

# end
