# Histogram with Box Plot [P292]

# Libraries
import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.stats import gaussian_kde



# Insights, improvements and bugfix
# 01 - Extend KDE line up to the zero (left and right margins),
# 02 - Remove labels from boxplot and move xticks for upper part, to use
#          the same labels for boh plots,
# 03 - Create a vertical version for plot (better usage if paper),
# 04 - Extend kde line up to zero (left and right margins),
#


# ----------------------------------------------------------------------
def plot_histbox(data, title=None, xlabel=None, bins="sqrt",
                 kde=True, meanline=True, medianline=True, notch="auto",
                 grid_axes="y", linebehind=True, tail_size=15,
                 savefig=False, verbose=True):
    """
    Plots the histogram of a given **data**.

    Variables:
    * data: Pandas Series, Numpy array or Python list,
    * title: Title for the plot (default="Histogram - {column name}"),
    * xlabel: Label for x_axis (default=None).
    * bins: Number of bins for plot (default="sqrt"). Check *binning*
            module for more details.
    * kde: Plot the Kernel-Gaussian density estimation line,
    * meanline: Plot a green line showing the mean,
    * medianline: Plot an orange line showing the median,
    * grid_axes: Plot axis (dafault=y),
    * linebehind = Plots mean and median line behind the plot,
    * savefig: True or False*. If True will save a report with the title
               name and do not show the plot. If False will not save the
               report but will show in the screen.(default=False),
    * verbose: True* or False (quiet mode). If True will print some in-
               formation about the data analysis and plot (default=True),
     
    """
    # Data preparation
    data = np.array(data)
    data = data[~(np.isnan(data))]          # Remove NaNs
    
    # Title
    if(title == None):
        title = "Histogram with Boxplot"

    
    # Colors
    colors = {"blue": "navy",
              "red": "darkred",
              "orange": "orange",
              "green": "darkgreen"}

    # Bins
    # more info: https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html
    bins_list = ["fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"]

    if(isinstance(bins, int) == True):
        no_bins = bins

    elif(bins_list.count(bins) == 1):
        no_bins = np.histogram_bin_edges(data, bins=bins).size

    elif(bins == "min"):
        no_bins = np.min([np.histogram_bin_edges(data, bins=x).size for x in bins_list])

    elif(bins == "max"):
        no_bins = np.max([np.histogram_bin_edges(data, bins=x).size for x in bins_list])

    elif(bins == "median"):
        no_bins = int(np.median([np.histogram_bin_edges(diff, bins=x).size for x in bins_list]))

    else:
        print(f' >>> Error: "bins" option not valid. Using "sqrt" as forced option')
        no_bins = np.histogram_bin_edges(data, bins="sqrt").size


    # Grid Axis
    grid_list = ["x", "y", "both"]
    if(grid_list.count(grid_axes) == 0):
        grid_axes = "y"
        print(f' >>> Error: "grid_axes" oprtion not valid. Using "y" as forced option.')


    # Define Auto-Notch
    if(notch == True or notch == False or notch == "auto"):
        pass

    else:
        notch == "auto"


    if(notch == "auto"):
        std_error = np.std(data) / np.sqrt(np.size(data))
        ci = stats.norm.interval(confidence=0.95, loc=np.median(data), scale=std_error)
        ci_lower, ci_upper = ci[0], ci[1]

        q1 = np.percentile(data, q=25)
        q3 = np.percentile(data, q=75)

        if(ci_lower <= q1 or ci_upper >= q3):
            notch = False

        else:
            notch = True
           
    # Histogram settings 
    # KDE: Kernel-density estimate for gaussian distribution
    if(kde == True):
        bins_alpha = 0.7
        bins_edge = colors["blue"]
        density = True
        ylabel = "density"

        # Add tail for the density line
        x_min = data.min()
        x_max = data.max()
        step = (x_max - x_min) * (tail_size / 100)      # Improvement No.01
        
        kde_space = np.linspace(start=(x_min - step), stop=(x_max + step), num=(50 * no_bins))
        kde_line = gaussian_kde(data, weights=None)(kde_space)

    else:
        bins_alpha = 1
        bins_edge = "dimgrey"
        density = False
        ylabel = "frequency"


    # RC Params
    set_rcparams()
    

    # Plot
    fig = plt.figure(figsize=[6, 3.375])        # Widescreen [16:9]
    grd = fig.add_gridspec(ncols=1, width_ratios=[1],
                           nrows=2, height_ratios=[7, 3])

    ax0 = fig.add_subplot(grd[0, 0])                # Histogram
    ax1 = fig.add_subplot(grd[1, 0], sharex=ax0)    # Boxplot
    
    fig.suptitle(title, fontsize=10, fontweight="bold", x=0.98, ha="right")

    # Histogram
    ax0.hist(data, bins=no_bins, density=density, color=colors["blue"],
             alpha=bins_alpha, edgecolor=bins_edge, zorder=20)

    ax0.grid(axis=grid_axes, color="lightgrey", linestyle="--", linewidth=0.5, zorder=10)

    # Density line
    if(kde == True):
        ax0.plot(kde_space, kde_line, color=colors["red"], linewidth=1.5, label="kde", zorder=23)

    if(linebehind == True): 
        zorder = 11
    
    else: 
        zorder = 21

    # Mean Lines
    if(meanline == True):
        ax0.axvline(x=np.mean(data), color=colors["green"], linewidth=1.0, label="mean", zorder=zorder)

    if(medianline == True):
        ax0.axvline(x=np.median(data), color=colors["orange"], linewidth=1.0, label="median", zorder=zorder)


    # Labels
    if(xlabel != None):
        ax0.set_xlabel(xlabel, loc="right")

    if(ylabel != None):
        ax0.set_ylabel(ylabel, loc="top")


    if(kde == True or meanline == True or medianline == True):
        ax0.legend(fontsize=9, loc="upper right", framealpha=1).set_zorder(99)


    # Boxplot
    # Parameters
    boxprops = dict(color="black", linestyle="-", linewidth=1.5)
    whiskerprops = dict(color="black", linestyle="-", linewidth=1.5)
    capprops = dict(color="black", linestyle="-", linewidth=1.5)
    medianprops = dict(color="orange", linestyle="-", linewidth=1.5)
    flierprops = dict(markerfacecolor="darkred", markeredgecolor="black", marker="o", markersize=6)

    ax1.boxplot(data, vert=False, widths=[0.6], notch=notch, boxprops=boxprops, whiskerprops=whiskerprops,
                     medianprops=medianprops, capprops=capprops, flierprops=flierprops, zorder=20)

        
    ax1.grid(axis="x", color="lightgrey", linestyle="--", linewidth=0.5, zorder=10)

    ax1.set_yticks([])
    ax1.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=False)
    

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

