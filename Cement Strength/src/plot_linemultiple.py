# Plot Line [P264] -----------------------------------------------------

# Versions
# 01 - Jan 11th, 2024 - Starter
#


# Insights, improvements and bugfix
# 01 - Limit up to 05 (five) columns


# Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# -----------------------------------------------------------------------

def plot_linemultiple(DataFrame, columns=None, title=None, xlabel=None,
                      ylabel=None, linewidth=1.5, grid="both", remove_axis=False,
                      legend_loc="best", savefig=False, verbose=True):

    """
    Plots a multiple line graph with **x** and **y**.
    Objective is to make life easier for a first simple plot and be a
    support for more detailed graphs.

    """
    # Data preparation
    data = DataFrame.copy()
    columns = col_select(data, columns)         # Improvement #01

    # Data separation
    data = data[columns]
    #data = data.dropna().reset_index(drop=True)

    # Title
    if(title == None):
        title = "Line multiple plot"

    # Grid (Default mode is **both**)
    grid_options = ["both", "y", "x"]
    if(grid_options.count(grid) == 0):
        grid = "both"

    # Colors
    colors = ["navy", "darkred", "orange", "darkgreen", "darkviolet"]
    colors = colors[0:len(columns)]


    # RC Params
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 8
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["ps.papersize"] = "A4"
    plt.rcParams["xtick.direction"] = "inout"
    plt.rcParams["ytick.direction"] = "inout"
    plt.rcParams["xtick.major.size"] = 3.5
    plt.rcParams["ytick.major.size"] = 3.5

    # Plot
    fig = plt.figure(figsize=[6, 3.375])        # Widescreen [16:9]
    fig.suptitle(title, fontsize=10, fontweight="bold", x=0.98, ha="right")
    ax = plt.axes()

    for col, color in zip(columns, colors):
        x = np.array(data[col].index)
        y = np.array(data[col])

        plt.plot(x, y, color=color, linewidth=linewidth, label=col, zorder=20)

    plt.grid(axis=grid, color="lightgrey", linestyle="--", linewidth=0.5, zorder=10)

    if(xlabel != None):
        plt.xlabel(xlabel, loc="right")

    if(ylabel != None):
        plt.ylabel(ylabel, loc="top")

    if(remove_axis == True):
        plt.tick_params(length=0,labelleft="on", labelbottom="on")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.left.set_visible(False)

    if(len(columns) > 1):
        plt.legend(loc=legend_loc, framealpha=1).set_zorder(99)


    # Printing
    plt.tight_layout()

    if(savefig == True):
        plt.savefig(title, dpi=320)
        if(verbose == True):
            print(f' > saved plot as "{title}.png"')

    else:
        plt.show()


    plt.close(fig)   

    return None


def col_select(DataFrame, columns):
    """
    Columns names verification.
    Also standatize the output as a list for pandas standard.
    
    """
    def column_checker(DataFrame, col_list):
        col_select = list()
        df_cols = DataFrame.columns.to_list()

        for i in col_list:
            if(df_cols.count(i) == 1):
                col_select.append(i)


        return col_select


    # Columns preparation
    if(columns == "all"):
        # Default: takes **all** columns from DataFrame.
        col_select = DataFrame.columns.to_list()

    elif(isinstance(columns, str) == True):
        # Tranforms a sting into a list
        columns = columns.replace(" ", "")
        columns = columns.split(",")
        col_select = column_checker(DataFrame, columns)

    elif(isinstance(columns, list) == True):
        col_select = column_checker(DataFrame, columns)

    else:
        col_select = list()


    return col_select


# end
