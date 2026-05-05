# activation_functions.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

import numpy as np
from . import backend as BE


# Activation functions and their derivatives
def sigmoid(x):
    """
    Sigmoid activation function.\n
    Parameter
    ---------
    :x: Array.
    """
    output = 1/(1+BE.xp.exp(-x))
    return output

def sigmoid_der(f):
    """
    Derivative of Sigmoid activation function.\n
    Parameter
    ---------
    :f: Activated output.
    """
    return f * (1 - f)

def tanh(x):
    """
    Tanh activation function.\n
    Parameter
    ---------
    :x: Array.
    """
    return BE.xp.tanh(x)

def tanh_der(f):
    """
    Derivative of Tanh activation function.\n
    Parameter
    ---------
    :f: Activated output.
    """
    return 1 - f**2

def reLU(x):
    """
    ReLU activation function: if less than 0, then 0.\n
    Parameter
    ---------
    :x: Array.
    """
    return BE.xp.maximum(0, x)

def reLU_der(f):
    """
    Derivative of reLU activation function.\n
    Parameter
    ---------
    :f: Activated output.
    """
    return BE.xp.where(f > 0, 1, 0)

def softmax(x):
    """
    Softmax activation function: ideal for multi-class classification.\n
    Parameter
    ---------
    :x: Array.
    """
    exp_x = BE.xp.exp(x - BE.xp.max(x, axis=1, keepdims=True))
    return exp_x / BE.xp.sum(exp_x, axis=1, keepdims=True)

def softmax_der(f):
    """
    Derivative of softmax activation function (Jacobian).\n
    Parameter
    ---------
    :f: Activated output.
    """
    return BE.xp.diagflat(f) - BE.xp.dot(f, f.T)

def linear(x):
    """
    Linear activation function: ideal for regression.\n
    Parameter
    ---------
    :x: Array.
    """
    return x

def linear_der(f):
    """
    Derivative of linear activation function.\n
    Parameter
    ---------
    :f: Activated output.
    """
    return BE.xp.ones_like(f)