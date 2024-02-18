# AutoMPG [P316]

# Libraries
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as st

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# Personal modules
sys.path.append(r"c:\python_modules")

from autompg_tools import *
from plot_histbox import *


# Program --------------------------------------------------------------
df = load_dataset()
df = df.drop(columns=["car_name"])
df = df.drop(columns=["model_year", "origin"])

# Data preparation
x, y = target_split(df, target="kpl")

np.random.seed(137)
seed_list = np.random.randint(low=0, high=1000, size=50)

# Model with different seed. Create a pool
df_results = pd.DataFrame(data=[])

for seed in seed_list:
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.65, random_state=seed)
    x_train, x_test = scaler(x_train, x_test, method=StandardScaler())
    _, results = regr_linregr(x_train, x_test, y_train, y_test)

    df_results = store_results(df_results, results)


# Pool performance
for col in df_results.columns:
    plot_histbox(df_results[col], title=f"AutoMPG - 11a - LinRegr with seed control - {col}", savefig=True)

    mean = df_results[col].mean()
    stddev = df_results[col].std()
    print(f" > {col}: {mean:.4f} ± {stddev:.4f}")


# T-Test (one sample)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html

mean_ref = 0.87212
# Mean from last experiment with a single seed (331)

ttest = st.ttest_1samp(df_results["pearson"], popmean=mean_ref, alternative="two-sided").pvalue
print(f" > T-Test: {ttest:.5f}, popmean={mean_ref}")









# end

