# gradients.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

from . import backend as BE

# Stochastic Gradient Descent
class SGD:
    """
    Updates the parameteres using only one randomly chosen data point for each step.
    Introduces stochasticness.
    Beneficial for:
        - Computational efficiency
        - Faster convergence
        - Escape local minima
    """
    def __init__(self, learning_rate = 0.01):
        """
        Initialization of the Stochastic Gradient Descent.\n
        Parameters
        ----------
        :learning_rate: Hyperparameter that controls the step size.\n
        """
        self.learning_rate = learning_rate
    
    def clip(self, grad, clip_value=1.0):
        norm = BE.xp.linalg.norm(grad)
        if norm > clip_value:
            grad *= clip_value / norm
        return grad

    def update(self, layer, wd = 1e-5, clipping = False):
        """
        Updates the layer's weights and biases via dtored gradients.\n
        Parameters
        ----------
        :layer: Dense layer that is currently being computed
        """


        if layer.d_weights is not None and layer.d_biases is not None:
            if clipping:
                layer.d_weights = self.clip(layer.d_weights)
                layer.d_biases = self.clip(layer.d_biases)
            layer.d_weights += 2 * wd * layer.weights
            layer.weights -= self.learning_rate * layer.d_weights
            layer.biases -= self.learning_rate * layer.d_biases
        else:
            import warnings
            warnings.warn("Gradients not found for a layer. Skipping update.")