# Scatter Plot with Histograms [P287] ----------------------------------

# Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Versions
# 01 - Jan 01st, 2021 - Starter
# 02 - Jan 30th, 2021 - Add labels into axis
# 03 - Feb 03rd, 2021 - Adjust figsize for best print (A4 Ratio)
# 04 - Jan 11th, 2024 - Refactoring
#

# Insights, improvements and bugfix
# 01 - Add bins_select for better vizualization
# 02 -


# -----------------------------------------------------------------------

def plot_scatterhist(x, y, title=None, xlabel=None, ylabel=None, color="navy",
                       alpha=0.8, mark_size=20, bins="sqrt",
                       savefig=False, verbose=True):
    """

    More info:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

    """
    # Data preparation
    x = np.array(x)
    y = np.array(y)

    # Title
    if(title == None):
        title = "Scatter with Histogram"

    # Bins
    # more info: https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html
    bins_list = ["fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"]

    if(isinstance(bins, int) == True):
        bins_x = bins
        bins_y = bins

    elif(bins_list.count(bins) == 1):
        bins_x = np.histogram_bin_edges(x, bins=bins).size
        bins_y = np.histogram_bin_edges(y, bins=bins).size

    elif(bins == "min"):
        bins_x = np.min([np.histogram_bin_edges(x, bins=s).size for s in bins_list])
        bins_y = np.min([np.histogram_bin_edges(y, bins=s).size for s in bins_list])

    elif(bins == "max"):
        bins_x = np.max([np.histogram_bin_edges(x, bins=s).size for s in bins_list])
        bins_y = np.max([np.histogram_bin_edges(y, bins=s).size for s in bins_list])

    elif(bins == "median"):
        bins_x = int(np.median([np.histogram_bin_edges(x, bins=s).size for s in bins_list]))
        bins_y = int(np.median([np.histogram_bin_edges(y, bins=s).size for s in bins_list]))
       
    else:
        print(f' >>> Error: "bins" option not valid. Using "sqrt" as forced option')
        bins = "sqrt"
        bins_x = np.histogram_bin_edges(x, bins=bins).size
        bins_y = np.histogram_bin_edges(y, bins=bins).size

    # RC Params
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["ps.papersize"] = "A4"
    plt.rcParams["xtick.direction"] = "inout"
    plt.rcParams["ytick.direction"] = "inout"
    plt.rcParams["xtick.major.size"] = 3.5
    plt.rcParams["ytick.major.size"] = 3.5

    # Plot
    # Figure
    fig = plt.figure(figsize=[6, 3.375])    # Widescreen 16:9
    grd = fig.add_gridspec(ncols=2, width_ratios=[3, 1],
                           nrows=2, height_ratios=[1, 3])

    ax0 = fig.add_subplot(grd[1, 0])                # Scatter (main plot)
    ax1 = fig.add_subplot(grd[0, 0], sharex=ax0)    # Histogram for x
    ax2 = fig.add_subplot(grd[1, 1], sharey=ax0)    # Histogram for y

    fig.suptitle(title, fontsize=10, fontweight="bold", x=0.98, ha="right")

    # Scatter plot
    ax0.scatter(x, y, s=mark_size, color=color, edgecolor="white", alpha=alpha, zorder=20)
    ax0.grid(axis="both", color="grey", linestyle="--", linewidth=0.5, zorder=5)       


    if(xlabel != None):
        ax0.set_xlabel(xlabel, loc="center")

    if(ylabel != None):
        ax0.set_ylabel(ylabel, loc="center")


    # Histogram for x
    ax1.hist(x, bins=bins_x, color=color, orientation="vertical", zorder=20)
    ax1.grid(axis="both", color="grey", linestyle="--", linewidth=0.5, zorder=5)       
    ax1.tick_params(axis="x", labelbottom=False)


    # Histogram for y
    ax2.hist(y, bins=bins_y, color=color, orientation="horizontal", zorder=20)
    ax2.grid(axis="both", color="grey", linestyle="--", linewidth=0.5, zorder=5)       
    ax2.tick_params(axis="y", labelleft=False)
    

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


# end
