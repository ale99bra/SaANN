# layers.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

import numpy as np
from . import activation_functions as AF
from . import initiations as In

# Dense Layer class
class DenseLayer:
    """
    Dense Layer is a layer in which all neurons recieve the output of each
    previous neuron.
        - Fully connected
        - Each connection has a learnable weight
        - Each neuron has a bias parameter
    """

    def __init__(self, num_inputs, num_neurons, activation_function, init_function, random_scale = 0.01):
        """
        Initialize the dense layer.\n
        Parameters
        ----------
        :num_imputs: Number of input features from the previous layer or initial input.\n
        :num_neurons: Number of neurons in the current dense layer.\n
        :activation_function: Activation function for the non-linearity: 'sigmoid', 'relu', 'tanh', 'softmax' or 'linear'.\n
        :init_function: Initialization function for the weigths: 'random', 'xavier' (or 'glorot') or 'he' (or 'kaiming').\n
        :random_scale: Scale for the 'random' initialization.
        """
        
        #self.weights = np.random.rand(num_inputs, num_neurons)*1e-1 #the size is a 2D matrix of the inputs and the nuerons
        self.biases = np.zeros((1, num_neurons))#a biases VECTOR

        if init_function == "random":
            self.weights = In.random_scaled_init(num_inputs=num_inputs, num_neurons=num_neurons, scale=random_scale)
        elif init_function in ("xavier", "glorot"):
            self.weights = In.xavier_init(num_inputs=num_inputs, num_neurons=num_neurons)
        elif init_function in ("he", "kaiming"):
            self.weights = In.he_init(num_inputs=num_inputs, num_neurons=num_neurons)
        else:
            raise ValueError(f"Initialization function '{init_function}' not found.\nPlease choose: 'random', 'xavier' or 'he'.")

        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.activation = activation_function

        """
        NOTE: we are now working with 2D vectors (matrix). 
        The weights matrix is organized:
            - columns: weights for each neuron
            - rows: connection of each input to all neurons

        The biases vection is 1D: is a row vector with one bias per neuron
        """

    def forward(self, inputs):
        """
        Performs the forwards pass through the dense layer.\n
        Parameters
        ----------
        :input: Array of input data. Shape must match the weights' shape
        """
        if inputs.shape[1] != self.weights.shape[0]: #if shapes do not match, raise the error
            raise ValueError(f"The size of the input ({inputs.shape[1]}) needs to match the neuron's input size ({self.weights.shape[0]})")
        
        self.inputs = inputs

        self.weighted_sum = np.dot(self.inputs, self.weights)

        raw_output = self.weighted_sum + self.biases

        if self.activation == "sigmoid":
            self.output = AF.sigmoid(raw_output)
            self.d_output = AF.sigmoid_der(self.output)
        elif self.activation == "relu":
            self.output = AF.reLU(raw_output)
            self.d_output = AF.reLU_der(self.output)
        elif self.activation == "softmax":
            self.output = AF.softmax(raw_output)
            self.d_output = None
        elif self.activation == "linear":
            self.output = AF.linear(raw_output)
            self.d_output = AF.linear_der(self.output)
        elif self.activation == "tanh":
            self.output = AF.tanh(raw_output)
            self.d_output = AF.tanh_der(self.output)
        else:
            raise ValueError(f"Activation function '{self.activation}' not found.\nPlease choose: 'sigmoid', 'relu', 'tanh', 'softmax' or 'linear'.")

        return self.output
    
    def backward(self, d_loss_wrt_output, wd):
        """
        Calculating the backpropagation.\n
        Parameters
        ----------
        :d_loss_wrt_output: Gradient of the loss with respect to this layer's output
        """
        if self.activation == "softmax":
            d_activation_output = d_loss_wrt_output
        else:
            d_activation_output = d_loss_wrt_output * self.d_output #gradient of loss w.r.t. pre-activation output

        batch_size = self.inputs.shape[0]

        self.d_weights = np.dot(self.inputs.T, d_activation_output) / batch_size # gradient of loss w.r.t. weights
        self.d_biases = np.sum(d_activation_output, axis=0, keepdims=True) / batch_size # gradient of loss w.r.t. biases

        self.d_weights += 2 * wd * self.weights

        self.d_weights = np.clip(self.d_weights, -1, 1)
        self.d_biases = np.clip(self.d_biases, -1, 1)

        d_loss_wrt_prev_output = np.dot(d_activation_output, self.weights.T)

        return d_loss_wrt_prev_output

# Multi-Layer Perceptron class
class MLP:
    """
    Multi-Layer Perceptron (MLP): NN architecture consisting of multiple sequencial dense layers 
        - input layer (raw data)
        - 1+ hidden layers (intermediate computations)
        - output layer (final results)
    """
    def __init__(self, layers_info):
        """
        Initialize the Multi-Layer Perceptron.\n
        Parameters
        ----------
        :layer_info: Information regarding how to construct the dense layers. Needs to be in the format of a list:\n[(#inputs, #neurons, 'activation function', 'initiation function'), ...].\n
        """
        self.layers = []

        self.add_layers(layers_info)

    def add_layers(self, layers_info):
        """
        Adds layers to the MLP architecture.\n
        Parameters
        ---------
        :layers_info: Information regarding how to construct the dense layers. Needs to be in the format of a list:\n[(#inputs, #neurons, 'activation function', 'initiation function'), ...].
        """

        #Checks
        if len(layers_info[0]) != 4:
            raise ValueError(f"Shape of 'layer_info' ({len(layers_info), len(layers_info[0])}) is not proper. Needs to be (x, 3)\nEach row needs to be (#inputs, #neurons, activation function).")
        for i in range(len(layers_info)):
            if i == len(layers_info)-1:
                continue
            if layers_info[i+1][0] != layers_info[i][1]:
                raise ValueError(f"The number of inputs for layer (apart from the input layer) needs to reflect the number of neurons from the previous layer.\n Please check layer #{i+1}.")

        #Create layers
        for n_inputs, n_neurons, activation, initialization in layers_info:
            self.layers.append(DenseLayer(num_inputs=n_inputs, num_neurons=n_neurons, activation_function=activation.lower(), init_function=initialization.lower()))

    def forward(self, inputs):
        """
        Performs the forward propagation through each layer of the MLP\n
        Parameters
        ----------
        :inputs: Input data for the input layer
        """
        curr_input = inputs #initialization

        for layer in self.layers: #iterate through each layer
            curr_input = layer.forward(curr_input) #update the input
        return curr_input
    
    def backward(self, d_loss_wrt_pred, wd):
        """
        Performs the backpropagation.\n
        Parameters
        ----------
        :d_loss_wrt_pred: gradient of loss w.r.t. the final output
        """
        curr_d_loss = d_loss_wrt_pred
        for layer in reversed(self.layers):
            curr_d_loss = layer.backward(curr_d_loss, wd)

        return curr_d_loss