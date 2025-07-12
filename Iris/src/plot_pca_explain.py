# PCA Explainability [P521] ---------------------------------------------

# Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# -----------------------------------------------------------------------
def plot_pca_explain(array, title=None, color_bar="navy", color_line="darkred",
                     grid="y", threshold=95, remove_axis=False, xrotation=False,
                     savefig=False, verbose=True):
    """


    """
    # Data preparation
    array = np.array(array)
    array = array * 100    
    cumulative = np.cumsum(array)
    labels = [f"PC{i}" for i in range(1, (array.size + 1))]

    # Title
    if(title == None):
        title = "PCA Explainability"

    # Grid Axis
    grid_default = "y"
    grid_list = ["x", "y", "both"]
    if(grid_list.count(grid) == 0):
        print(f' >>> Error: "grid" option not valid. Using "{grid_default}" as forced option.')
        grid = grid_default[:]

    # RC Params
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 8
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["ps.papersize"] = "A4"
    plt.rcParams["xtick.direction"] = "inout"
    plt.rcParams["ytick.direction"] = "inout"
    plt.rcParams["xtick.major.size"] = 0
    plt.rcParams["ytick.major.size"] = 3.5


    # Plot
    fig = plt.figure(figsize=[6, 3.375])        # Widescreen [16:9]
    fig.suptitle(title, fontsize=10, fontweight="bold", x=0.98, ha="right")
    ax = plt.axes()

    plt.bar(labels, height=array, color=color_bar, edgecolor="black", label="Explained variance",zorder=20)
    plt.plot(labels, cumulative, color=color_line, label="Cumulative explainance", zorder=21)
    plt.axhline(y=threshold, color="darkgreen", linestyle="--", linewidth=0.5, label=f"threshold {threshold}%", zorder=19)

    plt.grid(axis=grid, color="lightgrey", linestyle="--", linewidth=0.5, zorder=10)
    plt.ylabel("%", loc="top")

    #plt.legend(loc="lower left", framealpha=1).set_zorder(99)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3)

    if(xrotation == True):
        plt.xticks(rotation=90)       

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

