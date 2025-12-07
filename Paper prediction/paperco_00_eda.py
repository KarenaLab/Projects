# Project Paper CO

# Libraries
import os
import itertools
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt


# Project libraries
from src.paper_co_tools import (load_dataset, remove_cols_unique, feat_eng_runtime_inv,
                                organize_report)

from src.plot_histbox import plot_histbox
from src.plot_barv import plot_barv
from src.plot_scatterhist import plot_scatterhist
from src.plot_heatmap import plot_heatmap
from src.plot_line import plot_line


# Functions
def check_failure_data(DataFrame):
    """
    Function to check IF the **asset_id** and **runtime** orchestration
    is working corretly. When runtime stops to increase, it means the machine
    had a failure and asset_id need to be increased.

    If working correctly, asset_id increase every time runtime resets for 1 (one).

    """
    # "asset_id" and "runtime".
    # **asset_id** representes a complete run of the asset until its failure,
    # after the failure, the asset_id value is increased.
    # **runtime** is a measure of time that resets after failure.
    #
    # So, every asset_id increase, runtime resets for 1.
    
    errors = list()
    index = list(DataFrame.index)[0:-1]

    for i in index:
        asset_t0 = df.loc[i, "asset_id"]
        runtime_t0 = df.loc[i, "runtime"]

        asset_t1 = df.loc[i+1, "asset_id"]
        runtime_t1 = df.loc[i+1, "runtime"]

        if(asset_t1 > asset_t0):
            if(runtime_t1 != 1):
                errors.append(asset_t0)

    return errors


def plot_columns_nunique(DataFrame, title=None, savefig=False):
    """
    Function to check columns with a single/unique value.
    Data that could be important for process control but not relevant for
    machine learning models.

    Arguments:
    * DataFrame: Pandas DataFrame
    * title: Title for the plot and filename if savefig is enabled,
    * savefig: True or False* Selection to show or save the plot (default=False),

    Return:
    *** Shows or save a figure with data processing information.

    """
    # Data processing: Number of unique values per column
    cols_unique = pd.DataFrame(data=[])
    for col in df.columns:
        cols_unique.loc[col, "unique"] = df[col].nunique()

    # Plot
    plot_barv(x=cols_unique.index, height=cols_unique["unique"],
              title=title, xrotation=True, upside_down=False, savefig=savefig)

    return None


def cols_tag():
    cols = ["tag_2", "tag_3", "tag_4", "tag_6", "tag_7", "tag_8", "tag_9",
            "tag_11", "tag_12", "tag_13", "tag_14", "tag_15", "tag_17",
            "tag_20", "tag_21"]

    return cols

    
# Setup/Config ----------------------------------------------------------
path_main = os.getcwd()
path_database = os.path.join(path_main, "database")
path_report = os.path.join(path_main, "report")

SAVEFIG = True

warnings.filterwarnings("ignore")


# Program ---------------------------------------------------------------

# Import dataset for EDA
df = load_dataset(filename="pm_train.txt", path=path_database)

# Asset_id and runtime data validation
errors = check_failure_data(df)

# DataFrame preparation:
# >>> Check/Remove columns with unique values
#plot_columns_nunique(df, title=f"Paper CO - Unique values per variable", savefig=SAVEFIG)
df = remove_cols_unique(df, verbose=False)

# >>> Feature Engineering with runtime as countdown to failure
df = feat_eng_runtime_inv(df)


# EDA Plots
# Due dataprocessing visualization ONLY, using a slice of 25% of data
df_sample = df.sample(frac=.25, random_state=314)

# >>> Univariate analysis
for col in df.columns:
    #plot_histbox(data=df_sample[col], title=f"Paper CO - Histogram - {col}", savefig=SAVEFIG)
    pass

# >>> Bivariate analysis
cols_comb = list(itertools.combinations(list(df.columns), 2))
for (var_x, var_y) in cols_comb:
    #plot_scatterhist(x=df_sample[var_x], y=df_sample[var_y], xlabel=var_x, ylabel=var_y, mark_size=10,
    #                 title=f"Paper CO - Scatter - {var_x} vs {var_y}", savefig=SAVEFIG)
    pass
    
#plot_heatmap(df_sample, title="Paper CO - Heatmap", savefig=SAVEFIG)

# >>> Asset ID
for i in df["asset_id"].unique():
    info = df.groupby(by="asset_id").get_group(i)

    for col in cols_tag():
        #plot_line(x=info.index, y=info[col], title=f"Paper CO - Tag line Asset {i} - {col}", savefig=True)
        pass
    
# >>> Time: Tags by the time (all together) - "Hairs" 
for col in cols_tag():
    title = f"Paper CO - Tag lines - {col}"
    fig = plt.figure(figsize=[6, 3.375])        # Widescreen [16:9]
    fig.suptitle(title, fontsize=10, fontweight="bold", x=0.98, ha="right")

    for i in df["asset_id"].unique():
        info = df.groupby(by="asset_id").get_group(i)
        plt.plot(info["runtime"], info[col], linewidth=0.3, zorder=20)

    plt.grid(axis="both", color="lightgrey", linestyle="--", linewidth=0.5, zorder=10)
    
    plt.tight_layout()
    #plt.savefig(title, dpi=320)
    #plt.show()

    plt.close(fig)

# >>> Time: Analysis of SMA Feature together
for col in cols_tag():
    title = f"Paper CO - Tag lines with smooth SMA - {col}"
    fig = plt.figure(figsize=[6, 3.375])        # Widescreen [16:9]
    fig.suptitle(title, fontsize=10, fontweight="bold", x=0.98, ha="right")

    for i in df["asset_id"].unique():
        info = df.groupby(by="asset_id").get_group(i)
        info[col] = info[col].rolling(window=5).mean()
        plt.plot(info["runtime"], info[col], linewidth=0.3, zorder=20)

    plt.grid(axis="both", color="lightgrey", linestyle="--", linewidth=0.5, zorder=10)
    
    plt.tight_layout()
    #plt.savefig(title, dpi=320)
    #plt.show()

    plt.close()



   

# Scout theme: "Always leave the campsite cleaner than you found it"
organize_report(src=path_main, dst="report")

