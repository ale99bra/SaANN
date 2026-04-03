# losses.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

import numpy as np

# Loss functions and their derivative
def MSE(y_true, y_pred):
    """
    Calculates the Mean Squared Error.\n
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    """
    return np.mean((y_true - y_pred)**2)

def MSE_der(y_true, y_pred):
    """
    Derivative of the MSE loss w.r.t. y_pred.\n
    Parameters
    ----------
    :y_true: Testing values array.
    :y_pred: Array of values predicted by the model.
    """
    return 2 * (y_pred - y_true) / y_true.shape[0] #normalized by the size

def MAE(y_true, y_pred):
    """
    Calculates the Mean Absolute Error.\n
    Parameters
    ----------
    :y_true: Testing values array.
    :y_pred: Array of values predicted by the model.
    """
    return np.mean(np.abs(y_true - y_pred))