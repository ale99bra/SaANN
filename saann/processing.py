# processing.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

import numpy as np
import pandas as pd
import warnings
from . import backend as BE


def train_test_split(X, y, split_test_percentage = 0.3):
    """
    Returns the X_train, X_test, y_train and y_test separated in regards to the split specified.
    
    Parameters
    ----------
    :param X: Features/input Ndarray.
    :param y: Target Ndarray.
    :param split_test_percentage: Size of the test array in percentage (0, 1) of the total size.
    
    Returns
    -------
    :return X_train: Array of inputs for training.
    :return X_test: Array of inputs for testing.
    :return y_train: Array of targets for training.
    :return y_test: Array of targets for testing.
    
    Raises
    ------
    :ValueError: If the split test percentage is not a float between 0 and 1, inclusive.\n
    :ValueError: If param X's (or y's) type is not in (list, BE.xp.ndarray, pd.DataFrame).\n
    :ValueError: If param X and param y have different lengths.
    
    Example
    -------
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, split_test_percentage = 0.3)
    """
    if isinstance(X, (list, BE.xp.ndarray, pd.DataFrame)):
        if isinstance(X, BE.xp.ndarray):
            pass
        else:
            try:
                tmp = type(X)
                X = BE.xp.array(X)
                warnings.warn(f"Converted 'X' ({tmp}) to {type(X)}.")
            except:
                raise TypeError(f"The given 'X' type ({type(X)}) is not supported.")
    else:
        raise TypeError(f"The given 'X' type ({type(X)}) is not supported.")
    
    if isinstance(y, (list, BE.xp.ndarray, pd.DataFrame)):
        if isinstance(y, BE.xp.ndarray): 
            pass
        else:
            try:
                tmp = type(y)
                y = BE.xp.array(y)
                warnings.warn(f"Converted 'y' ({tmp}) to {type(y)}.")
            except:
                raise TypeError(f"The given 'y' type ({type(y)}) is not supported.")
    else:
        raise TypeError(f"The given 'y' type ({type(y)}) is not supported.")
    if len(X) != len(y):
        raise ValueError(f"Lengths of X ({len(X)}) and y ({len(y)}) arrays do not match.")
    if split_test_percentage <= 0 or split_test_percentage >= 1:
        raise ValueError(f"The 'split_test_percentage' needs to be in the range (0, 1).")
    if split_test_percentage >= 0.5:
        warnings.warn("It is reccomended to use a 'split_test_percentage' of around 0.3.")

    X_train = X[0:int((1-split_test_percentage)*len(X))]
    y_train = y[0:int((1-split_test_percentage)*len(X))]
    X_test = X[int((1-split_test_percentage)*len(X)):]
    y_test = y[int((1-split_test_percentage)*len(X)):]

    try:
        y_train = y_train.reshape(-y.shape[1], y.shape[1])
        y_test = y_test.reshape(-y.shape[1], y.shape[1])
    except:
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    return X_train, X_test, y_train, y_test

class Scaling:
    """
    Class for the different scalings
    """

    def __init__(self):
        self.scaled_data = None

    def zScore(self, x):
        """
        Performs a z-score standardization.\n
        Parameter
        ---------
        :param x: Data array

        Return
        ------
        :return scaled_data: Data scaled via z-score standardization

        Example
        -------
            >>> scaler = Scaling()
        >>> x_scaled = scaler.zScore(x)
        """
        self.scaled_data = (x-BE.xp.mean(x))/(BE.xp.std(x))
        return self.scaled_data
    
    def MinMax(self, x):
        """
        Performs a Min-Max standardization.\n
        Parameters
        ----------
        :param x: Data array

        Return
        ------
        :return scaled_data: Data scaled via Min-Max standardization

        Example
        -------
            >>> scaler = Scaling()
        >>> x_scaled = scaler.MinMax(x)
        """
        self.scaled_data = (x - BE.xp.min(x))/(BE.xp.max(x)-BE.xp.min(x))
        return self.scaled_data
    
    def LogNorm(self, x):
        """
        Performs a natural log transformation.\n
        Parameters
        ----------
        :param x: Data array

        Return
        ------
        :return scaled_data: Data scaled via natural log normalization

        Example
        -------
            >>> scaler = Scaling()
        >>> x_scaled = scaler.LogNorm(x)
        """
        self.scaled_data = BE.xp.log1p(x)
        return self.scaled_data
    
    def MeanNorm(self, x):
        """
        Performs a mean normalization.\n
        Parameters
        ----------
        :param x: Data array

        Return
        ------
        :return scaled_data: Data scaled via mean normalization

        Example
        -------
            >>> scaler = Scaling()
        >>> x_scaled = scaler.MeanNorm(x)
        """
        self.scaled_data = (x-BE.xp.mean(x))/(BE.xp.max(x)-BE.xp.min(x))
        return self.scaled_data
