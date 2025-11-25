# Bar Horizontal [P354] -------------------------------------------------

# Insights, improvements and bugfix
# 01 - Add labels to the bars (Excel style)
# 02 - Create an ascendant sequence
# 03 - 


# Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# -----------------------------------------------------------------------

def plot_barh(x, width, title=None, xlabel=None, color="navy", left=0,
              upside_down=True, grid="x", remove_axis=False,
              savefig=False, verbose=True):
    """
    Plots a Horizontal bar. **width** is the **y** values and follows an
    **upside-down** sequence.

    Variables:
    * x: 
    * width:
    * title: Graph title (also the filename if verbose is True),
    * xlabel: Label for x axis,
    * color: Color of bars (default="navy"),
    * left: 
    * upside_down:
    * grid: Choose the grid to be plot (default="x"),
    * remove_axis: Remove axis and borders of graph (cleaner view),
    * savefig: Choose to save the plot as a .png figure (default=False),
    * verbose: Choose the verbose mode (default=False).
    
    """
    # Data preparation
    if(upside_down == True):
        x = np.flip(x, axis=0)          # np.flip to make plot upside-down
        width = np.flip(width, axis=0)
        
    else:
        x = np.array(x)
        width = np.array(width)


    # Title
    if(title == None):
        title = "Bar horizontal"


    # Grid Axis
    grid_default = "x"
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

    plt.barh(x, width=width, color=color, edgecolor="black", left=left, zorder=20)

    plt.grid(axis=grid, color="lightgrey", linestyle="--", linewidth=0.5, zorder=10)

    if(xlabel != None):
        plt.xlabel(xlabel, loc="right")

    if(remove_axis == True):
        plt.tick_params(length=0, labelleft="on", labelbottom="on")
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)


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

