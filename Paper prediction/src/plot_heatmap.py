# Heatmap [P262] -------------------------------------------------------

# Libraries
import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt


# Insights, improvements and bugfix
# Add rotation to x_axis labels (need to check anchor),
#


def plot_heatmap(DataFrame, columns="all", title=None, decimals=2,
                 method="pearson", colormap="Blues", fontsize=7,
                 remove_axis=True, savefig=False, verbose=True):
    """
    Plots a triangular **Heatmap** for **correlations** between variables,
    using Pearson, Spearman or Kendall methods.

    Variables:
    * DataFrame = Data to be plot,
    * title = Title for the plot (and the name for the file if savefig=True),
    * columns = "all" or a selected list of columns name,
    * decimals = Number of decimals to display (default=2),
    * method = Method for correlation: Pearson*, spearman or kendall,
      https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html
    * colormap = Blues*, Greys, Reds or Oranges (suggestion in reason of text match),
    * remove_axis = Remove top and right axis (cleaner viz),
    * savefig = True or False for saving the plot as a figure using title as name,
    * verbose = True* or False for messages.    

    """
    # Data Preparation
    data = DataFrame.copy()
    columns = col_select(data, columns)

    # Title
    if(title == None):
        title = f"Heatmap for correlation ({method})"

    else:
        title = f"{title} ({method})"


    # Columns preparation
    cols = list()
    for col in columns:
        if(is_numeric_dtype(data[col]) == True):
           cols.append(col)

    columns = cols[:]           # To avoid problems with duplicated variables
    data = data[columns]
    

    # Correlation (Heatmap)
    n_cols = len(columns)
    corr = data.corr(method=method).values
    corr = np.abs(np.round(corr, decimals=decimals))   

  
    # Removing Duplicated Data (Right Triangle Figure)
    for i in range(0, n_cols):
        for j in range(0, n_cols):
            if(i < j):
                corr[i, j] = 0

    # Figure size
    size_hor = 8
    size_ver = (size_hor * 9) / 16      # [16:9] Widescreen ratio (aspect)


    # RC Params
    set_rcparams(fontsize=fontsize)
   
    # Plot   
    fig = plt.figure(figsize=[size_hor, size_ver])
    fig.suptitle(title, fontsize=9, fontweight="bold", x=0.98, ha="right")

    ax = fig.add_subplot()
    im = ax.imshow(corr, cmap=colormap, aspect=(size_ver / size_hor))

    ax.set_yticks(np.arange(start=0, stop=n_cols), labels=columns, rotation=0, fontsize=fontsize)
    ax.set_xticks(np.arange(start=0, stop=n_cols), labels=columns, rotation=90, fontsize=fontsize)


    # Text Annotations
    for i in range(0, len(columns)):
        for j in range(0, len(columns)):
            if(i >= j):
                value = corr[i, j]

                if(value == 1): textcolor = "grey"                
                elif(value >= 0.6): textcolor = "white"
                else: textcolor = "black"

                text = ax.text(j, i, value, ha="center", va="center",
                               color=textcolor, fontsize=fontsize)

    # Cleaner viz
    if(remove_axis == True):
        plt.tick_params(length=0, labelleft="on", labelbottom="on")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)


    # Printing 
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


def set_rcparams(fontsize):
    # RC Params
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["ps.papersize"] = "A4"
    plt.rcParams["xtick.major.size"] = 0
    plt.rcParams["ytick.major.size"] = 0
    plt.rcParams["xtick.major.size"] = 3.5
    plt.rcParams["ytick.major.size"] = 3.5

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
