# Plot Line [P391] -----------------------------------------------------

# Insights, improvements and bugfix
# 


# Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# -----------------------------------------------------------------------
def plot_lineduo(x1, y1, y2, x2=None, title=None, label1=None, label2=None,
                 xlabel=None, ylabel=None, color1="navy", color2="darkred",
                 linewidth=1.5, grid="both", remove_axis=False, legend_loc="best",
                 savefig=False, verbose=True):

    """
    Plots a line graph with **x** and **y**.
    Objective is to make life easier for a first simple plot and be a
    support for more detailed graphs.

    """
    # Data preparation
    x1 = np.array(x1)
    y1 = np.array(y1)
    y2 = np.array(y2)

    if(x2 == None):
        x2 = x1

    else:
        x2 = np.array(x2)
    

    # Title
    if(title == None):
        title = "Line plot"

    # Grid Axis
    grid_default = "both"
    grid_list = ["x", "y", "both"]
    if(grid_list.count(grid) == 0):
        print(f' >>> Error: "grid" option not valid. Using "{grid_default}" as forced option.')
        grid = grid_default[:]


    # RC Params
    set_rcparams()
    

    # Plot
    fig = plt.figure(figsize=[6, 3.375])        # Widescreen [16:9]
    fig.suptitle(title, fontsize=10, fontweight="bold", x=0.98, ha="right")
    ax = plt.axes()

    plt.plot(x1, y1, color=color1, linewidth=linewidth, label=label1, zorder=20)
    plt.plot(x2, y2, color=color2, linewidth=linewidth, label=label2, zorder=19)

    plt.grid(axis=grid, color="lightgrey", linestyle="--", linewidth=0.5, zorder=10)

    if(xlabel != None):
        plt.xlabel(xlabel, loc="right")

    if(ylabel != None):
        plt.ylabel(ylabel, loc="top")

    if(label1 != None and label2 != None):
        plt.legend(loc=legend_loc, framealpha=1).set_zorder(99)

    if(remove_axis == True):
        plt.tick_params(length=0,labelleft="on", labelbottom="on")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.left.set_visible(False)


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


def set_rcparams():
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 8
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["ps.papersize"] = "A4"
    plt.rcParams["xtick.direction"] = "inout"
    plt.rcParams["ytick.direction"] = "inout"
    plt.rcParams["xtick.major.size"] = 3.5
    plt.rcParams["ytick.major.size"] = 0

    return None

