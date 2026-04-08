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
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    """
    return 2 * (y_pred - y_true) / y_true.shape[0] #normalized by the size

def MAE(y_true, y_pred):
    """
    Calculates the Mean Absolute Error.\n
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    """
    return np.mean(np.abs(y_true - y_pred))

def MAE_der(y_true, y_pred):
    """
    Calculates the Mean Absolute Error's gradient.\n
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    """
    return np.sign(y_pred - y_true) / y_true.shape[0]

def R2_score(y_true, y_pred):
    """
    Calculates the R-squared metric.
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    """

    res_SS = np.sum((y_true - y_pred)**2)
    tot_SS = np.sum((y_true - np.mean(y_true))**2)

    return 1 - res_SS/tot_SS

def Huber(y_true, y_pred, delta = 1):
    """
    Calculates the Huber loss.\n
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    :param delta: hyperparameter for defining the threshold - quadratic to linear
    """
    diff = y_true - y_pred
    quadratic = (diff**2)/2
    linear = delta * (np.abs(diff) - delta/2)
    
    score = np.mean(np.where(np.abs(diff) <= delta, quadratic, linear))

    return score

def Huber_der(y_true, y_pred, delta):
    """
    Calculates the Huber loss' gradient.\n
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    :param delta: hyperparameter for defining the threshold - quadratic to linear
    """
    diff = y_true - y_pred
    quadratic_der = -diff
    linear_der = -delta * np.sign(diff)
    
    score_der = np.mean(np.where(np.abs(diff) <= delta, quadratic_der, linear_der))

    return score_der

def cross_entropy(y_true, y_pred, epsilon=1e-12):
    """
    Calculates the cross_entropy loss.\n
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    """
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(y_true * np.log(y_pred), axis=1, keepdims=True)

def cross_entropy_der(y_true, y_pred):
    """
    Calculates the cross_entropy loss' gradient.\n
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    """
    return (y_pred - y_true)

if __name__ == "__main__":
    pred = np.linspace(0, 100, num = 26)
    true = np.linspace(0, 90, num = 26)

    r2_scoring = R2_score(true, pred)
    huber = Huber(true, pred, delta = 0.1)
    huber_der = Huber_der(true, pred, delta = 0.1)
    print(r2_scoring)
    print(huber)
    print(huber_der)