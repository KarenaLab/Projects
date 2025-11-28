# Bar Horizontal [P353] -------------------------------------------------

# Insights, improvements and bugfix
# 01 - Add labels to the bars (Excel style)
# 02 - Add rotation for x axis labels [Solved]
# 03 - 


# Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# -----------------------------------------------------------------------

def plot_barv(x, height, title=None, ylabel=None, color="navy",
              upside_down=True, grid="y", remove_axis=False, xrotation=False,
              savefig=False, verbose=True):
    """
    height = y
    upside-down sequence

    """
    # Data preparation
    if(upside_down == True):
        x = np.flip(x, axis=0)          # np.flip to make plot upside-down
        height = np.flip(height, axis=0)
        
    else:
        x = np.array(x)
        height = np.array(height)
        

    # Title
    if(title == None):
        title = "Bar vertical"


    # Grid Axis
    grid_default = "y"
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

    plt.bar(x, height=height, color=color, edgecolor="black", zorder=20)

    plt.grid(axis=grid, color="lightgrey", linestyle="--", linewidth=0.5, zorder=10)

    if(xrotation == True):
        plt.xticks(rotation=90)

    if(ylabel != None):
        plt.ylabel(ylabel, loc="top")

    if(remove_axis == True):
        plt.tick_params(length=0, labelleft="on", labelbottom="on")
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
    plt.rcParams["xtick.major.size"] = 0
    plt.rcParams["ytick.major.size"] = 3.5

    return None
