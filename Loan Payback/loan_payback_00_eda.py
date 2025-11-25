# [P531] Loan payback - S5_E11
# EDA

# Insights, improvements and bugfix
#


# Libraries
import os
import itertools

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt


# Personal modules
from src.loan_payback_tools import (load_dataset, organize_report,
                                    cols_categoric, cols_numeric)

from src.plot_histbox import plot_histbox
from src.plot_scatterhist import plot_scatterhist
from src.plot_heatmap import plot_heatmap
from src.plot_barh import plot_barh


# Functions
def brief_category(Series):
    data = np.array(Series)
    total = data.size

    counter = pd.DataFrame(data=[])
    for i in np.unique(data):
        count = (data == i).sum()
        pct = np.round((count / total) * 100, decimals=3)

        counter.loc[i, "count"] = count
        counter.loc[i, "pct"] = pct

    counter = counter.sort_values(by="count", ascending=False)


    return counter



# Setup/Config
savefig = True


# Program --------------------------------------------------------------
df = load_dataset(filename="train.csv", path=".\database")
target = "loan_paid_back"

sample = df.sample(n=10000, random_state=314)

"""
# Univariate analysis
for col in cols_numeric():
    plot_histbox(data=sample[col], title=f"Loan payback S5E11 - Histogram {col}",
                 savefig=savefig)

for col in cols_categoric():
    pass


# Bivariate analysis: Combination of all numerical cols
var_comb = list(itertools.combinations(cols_numeric(), 2))
for var_x, var_y in var_comb:
    plot_scatterhist(x=sample[var_x], xlabel=var_x, y=sample[var_y], ylabel=var_y,
                     mark_size=15, title=f"Loan payback S5E11 - ScatterHist {var_x} vs {var_y}",
                     savefig=savefig)


# Bivariave analysis: Numerical cols versus target:
for var_x in cols_numeric():
    plot_scatterhist(x=sample[var_x], xlabel=var_x, y=sample[target], ylabel=target,
                     mark_size=15, color="darkred", title=f"Loan payback S5E11 - ScatterHist {var_x} vs {target}",
                     savefig=savefig)


# Heatmap
plot_heatmap(sample, columns=cols_numeric(), title=f"Loan payback S5E11 - Heatmap",
             savefig=savefig)

"""
for col in cols_categoric():
    info = brief_category(df[col])
    plot_barh(x=info.index, width=info["count"], title=f"Loan payback S5E11 - Category {col}", savefig=savefig)
       
        



# Scout theme: "Always leave the campsite cleaner than you found it"
organize_report(verbose=True)

