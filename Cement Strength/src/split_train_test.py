# [P229] Train Validation and Test split (holdout)

# Libraries
import numpy as np
import pandas as pd


# Functions
def split_train_test(DataFrame, train_size=70, seed=None, reset_index=True,
                     show_source=False):
    """
    

    """
    # Train and Test preparation
    cut = int((DataFrame.shape[0] * (train_size / 100)) + .5)

    # Seed preparation
    if(seed != None):
        np.random.seed(seed)

    # Train and Test split
    index = np.array(DataFrame.index)
    np.random.shuffle(index)

    train_index = index[0:cut]
    test_index = index[cut: ]

    train = DataFrame.iloc[train_index].reset_index(drop=reset_index)
    test = DataFrame.iloc[test_index].reset_index(drop=reset_index)

    
    if(show_source == False):
        return train, test

    else:
        return train, test, [train_index, test_index]

