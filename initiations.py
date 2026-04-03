# initiations.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

import numpy as np

# Initialization functions
def random_scaled_init(num_inputs, num_neurons, scale=0.01):
    """
    Initialize the weights using a RNG scaled by a factor.\n
    Parameters
    ----------
    :num_inputs: Number of inputs of the layer.
    :num_neurons: Number of neurons of the layer.
    :scale: scale applied to weights
    """
    return np.random.randn(num_inputs, num_neurons) * scale

def xavier_init(num_inputs, num_neurons):
    """
    Initialize the weights using the Xavier or Glorot initialization.
        Normal distribution with stddev = sqrt(2/(#inputs + #neurons))
        - Ideal for Sigmoid or Tanh activations.\n
    Parameters
    ----------
    :num_inputs: Number of inputs of the layer.
    :num_neurons: Number of neurons of the layer.
    """
    stddev = np.sqrt(2/(num_inputs + num_neurons))

    return np.random.randn(num_inputs, num_neurons) * stddev

def he_init(num_inputs, num_neurons):
    """
    Initialize the weights using the He (or Kaiming) initialization
        Uniform distribution within bounds (sqrt(6/#inputs)$)
        - Ideal for ReLU activations.\n
    Parameters
    ----------
    :num_inputs: Number of inputs of the layer.
    :num_neurons: Number of neurons of the layer.
    """
    limit = np.sqrt(6/num_inputs)

    return np.random.uniform(-limit, limit, (num_inputs, num_neurons))