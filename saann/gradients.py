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

class AdamW:

    def __init__(self, params, learning_rate = 1e-3, beta1 = 0.9, beta2 = 0.999, wd = 0.0, eps = 1e-8):
        self.params = params
        self.lr = learning_rate
        self.B1 = beta1
        self.B2 = beta2
        self.eps = eps
        self.wd = wd
        
        #initialize moments
        self.m = {}
        self.v = {}

        for name, p in self.params.items():
            if name.startswith("d"): continue
            self.m[name] = BE.xp.zeros_like(p)
            self.v[name] = BE.xp.zeros_like(p)

        self.t = 0

    def step(self):

        self.t += 1

        for name, g in self.params.items():
            if not name.startswith("d"): continue
            param_name = name[1:]
            param = self.params[param_name]

            if self.wd > 0.0:
                param -= self.lr * self.wd * param

            #update bias for m
            self.m[param_name] = (self.B1 * self.m[param_name] + (1-self.B1) * g)

            #update bias for v
            self.v[param_name] = (self.B2 * self.v[param_name] + (1-self.B2) * g**2)

            #correct biases
            m_hat = self.m[param_name] / (1-self.B1**self.t)
            v_hat = self.v[param_name] / (1-self.B2**self.t)

            #update param
            param -= self.lr * m_hat/(BE.xp.sqrt(v_hat)+self.eps)
    
    def zero_grad(self):
        for name, p in self.params.items():
            if name.startswith("d"): p[...]=0