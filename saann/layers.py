# layers.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

import warnings
from . import activation_functions as AF
from . import initiations as In
from . import backend as BE

# Dense Layer class
class DenseLayer:
    """
    Dense Layer is a layer in which all neurons recieve the output of each
    previous neuron.
        - Fully connected
        - Each connection has a learnable weight
        - Each neuron has a bias parameter
    """

    def __init__(self, extras, num_inputs, num_neurons, activation_function, init_function, random_scale = 0.01):
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

        if extras[0]:
            self.batchnorm = BatchNorm(dim=num_neurons)
        else:
            self.batchnorm = No_Batch()
        
        if extras[1]:
            self.dropout = Dropout()
        else:
            self.dropout = No_Dropout()
        
        #self.weights = BE.xp.random.rand(num_inputs, num_neurons)*1e-1 #the size is a 2D matrix of the inputs and the nuerons
        self.biases = BE.xp.zeros((1, num_neurons))#a biases VECTOR

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

        self.weighted_sum = BE.xp.dot(self.inputs, self.weights)

        raw_output = self.weighted_sum + self.biases
        
        raw_output = self.batchnorm.forward(raw_output)

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

        if self.activation == "softmax":
            pass
        else:
            self.output = self.dropout.forward(self.output)

        return self.output
    
    def backward(self, d_loss_wrt_output, wd):
        """
        Calculating the backpropagation.\n
        Parameters
        ----------
        :d_loss_wrt_output: Gradient of the loss with respect to this layer's output
        """
        d_loss_wrt_output = self.dropout.backward(d_loss_wrt_output)

        if self.activation == "softmax":
            d_activation_output = d_loss_wrt_output
        else:
            d_activation_output = d_loss_wrt_output * self.d_output #gradient of loss w.r.t. pre-activation output
        
        d_activation_output = self.batchnorm.backward(d_activation_output)

        batch_size = self.inputs.shape[0]

        self.d_weights = BE.xp.dot(self.inputs.T, d_activation_output) / batch_size # gradient of loss w.r.t. weights
        self.d_biases = BE.xp.sum(d_activation_output, axis=0, keepdims=True) / batch_size # gradient of loss w.r.t. biases

        self.d_weights += 2 * wd * self.weights

        """self.d_weights = BE.xp.clip(self.d_weights, -1, 1)
        self.d_biases = BE.xp.clip(self.d_biases, -1, 1)"""

        d_loss_wrt_prev_output = BE.xp.dot(d_activation_output, self.weights.T)

        return d_loss_wrt_prev_output
    
class No_Batch:
    def __init__(self):
        pass

    def forward(self, X, training=True):
        return X
    
    def backward(self, d_out):
        return d_out
    
    def update(self, lr):
        pass

class No_Dropout:
    def __init__(self, p=0.5):
        pass

    def forward(self, X, training=True):
        return X
        
    def backward(self, d_out):
        return d_out
    
class BatchNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = BE.xp.ones((1, dim))
        self.beta  = BE.xp.zeros((1, dim))

        # Running stats (for inference)
        self.running_mean = BE.xp.zeros((1, dim))
        self.running_var  = BE.xp.ones((1, dim))

    def forward(self, X, training=True):
        if training:
            self.mu = BE.xp.mean(X, axis=0, keepdims=True)
            self.var = BE.xp.var(X, axis=0, keepdims=True)

            self.x_norm = (X - self.mu) / BE.xp.sqrt(self.var + self.eps)
            out = self.gamma * self.x_norm + self.beta

            # Update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mu
            self.running_var  = self.momentum * self.running_var  + (1 - self.momentum) * self.var
        else:
            # Use running stats at inference
            x_norm = (X - self.running_mean) / BE.xp.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta

        self.X = X
        return out
    
    def backward(self, d_out):
        N, D = d_out.shape

        # Gradients for gamma and beta
        self.d_gamma = BE.xp.sum(d_out * self.x_norm, axis=0, keepdims=True)
        self.d_beta  = BE.xp.sum(d_out, axis=0, keepdims=True)

        # Backprop through normalization
        dx_norm = d_out * self.gamma
        dvar = BE.xp.sum(dx_norm * (self.X - self.mu) * -0.5 * (self.var + self.eps)**(-3/2), axis=0)
        dmu = BE.xp.sum(dx_norm * -1 / BE.xp.sqrt(self.var + self.eps), axis=0) + dvar * BE.xp.mean(-2 * (self.X - self.mu), axis=0)

        dX = dx_norm / BE.xp.sqrt(self.var + self.eps) + dvar * 2 * (self.X - self.mu) / N + dmu / N

        return dX
    
    def update(self, lr):
        self.gamma -= lr * self.d_gamma
        self.beta  -= lr * self.d_beta

class Dropout:
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, X, training=True):
        if training:
            self.mask = (BE.xp.random.rand(*X.shape) > self.p) / (1 - self.p)
            return X * self.mask
        else:
            return X
        
    def backward(self, d_out):
        try:
            return d_out * self.mask
        except:
            return d_out

# Multi-Layer Perceptron class
class MLP:
    """
    Multi-Layer Perceptron (MLP): NN architecture consisting of multiple sequencial dense layers 
        - input layer (raw data)
        - 1+ hidden layers (intermediate computations)
        - output layer (final results)
    """
    def __init__(self, layers_info, batch_norm = False, dropout = False):
        """
        Initialize the Multi-Layer Perceptron.\n
        Parameters
        ----------
        :layer_info: Information regarding how to construct the dense layers. Needs to be in the format of a list:\n[(#inputs, #neurons, 'activation function', 'initiation function'), ...].\n
        """
        self.layers = []
        self.extras = [batch_norm, dropout]

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
            self.layers.append(DenseLayer(extras = self.extras, num_inputs=n_inputs, num_neurons=n_neurons, activation_function=activation.lower(), init_function=initialization.lower()))

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
    
class RNNLayer:

    def __init__(self, input_dim, hidden_dim, activation_function="tanh", init_function = "xavier", random_scale = 0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation_function

        # activation function
        if self.activation == "sigmoid":
            warnings.warn(message="The 'sigmoid' activation function tends to produce vanishing gradients. 'tanh' activation is recommended for RNN.", category=UserWarning)
            self.act_function = AF.sigmoid
            self.d_act = AF.sigmoid_der
        elif self.activation == "relu":
            warnings.warn(message="The 'relu' activation function tends to produce exploding gradients. 'tanh' activation is recommended for RNN.", category=UserWarning)
            self.act_function = AF.reLU
            self.d_act = AF.reLU_der
        elif self.activation == "softmax":
            raise ValueError("'softmax' activation not allowed. 'tanh' activation is recommended for RNN.")
        elif self.activation == "linear":
            warnings.warn(message="The 'linear' activation function tends to be undtable if initiation not suitable. 'tanh' activation is recommended for RNN.", category=UserWarning)
            self.act_function = AF.linear
            self.d_act = AF.linear_der
        elif self.activation == "tanh":
            self.act_function = AF.tanh
            self.d_act = AF.tanh_der
        else:
            raise ValueError(f"Activation function '{self.activation}' not found.\nPlease choose: 'sigmoid', 'relu', 'tanh' (recommended) or 'linear'.")

        # initialize weights
        if init_function == "random":
            warnings.warn(message="The 'random' initiation is unstable. 'xavier' initiation is recommended for RNN.", category=UserWarning)
            self.weights_xh = In.random_scaled_init(num_inputs=input_dim, num_neurons=hidden_dim, scale=random_scale)
            self.weights_hh = In.random_scaled_init(num_inputs=hidden_dim, num_neurons=hidden_dim, scale=random_scale)
        elif init_function in ("xavier", "glorot"):
            self.weights_xh = In.xavier_init(num_inputs=input_dim, num_neurons=hidden_dim)
            self.weights_hh = In.xavier_init(num_inputs=hidden_dim, num_neurons=hidden_dim)
        elif init_function in ("he", "kaiming"):
            warnings.warn(message="The 'he' initiation tends to explode. 'xavier' initiation is recommended for RNN.", category=UserWarning)
            self.weights_xh = In.he_init(num_inputs=input_dim, num_neurons=hidden_dim)
            self.weights_hh = In.he_init(num_inputs=hidden_dim, num_neurons=hidden_dim)
        else:
            raise ValueError(f"Initialization function '{init_function}' not found.\nPlease choose: 'random', 'xavier' (recommended) or 'he'.")

        self.biases_h  = BE.xp.zeros((1, hidden_dim))

        # Gradients
        self.d_Wxh = None
        self.d_Whh = None
        self.d_bh  = None

    def forward(self, X):
        # X: (batch, seq_len, input_dim)
        # return: h_seq (batch, seq_len, hidden_dim), cache
        batch, seq_len, _ = X.shape

        H = BE.xp.zeros((batch, seq_len, self.hidden_dim))
        h_prev = BE.xp.zeros((batch, self.hidden_dim))

        cache = {"X": X, "H": [], "h_prev": [], "Z": []}

        for t in range(seq_len):
            x_t = X[:, t, :]
            z_t = x_t @ self.weights_xh + h_prev @ self.weights_hh + self.biases_h
            h_t = self.act_function(z_t)
            
            H[:, t, :] = h_t

            cache["Z"].append(z_t)
            cache["H"].append(h_t)
            cache["h_prev"].append(h_prev)
            h_prev = h_t

        return H, cache

    def backward(self, dH, cache):
        X = cache["X"]
        H_list = cache["H"]
        h_prev_list = cache["h_prev"]

        batch, seq_len, _ = X.shape

        d_Wxh = BE.xp.zeros_like(self.weights_xh)
        d_Whh = BE.xp.zeros_like(self.weights_hh)
        d_bh  = BE.xp.zeros_like(self.biases_h)

        d_X = BE.xp.zeros_like(X)
        d_h_next = BE.xp.zeros((batch, self.hidden_dim))

        for t in reversed(range(seq_len)):
            h_t = H_list[t]
            h_prev = h_prev_list[t]

            # total gradient at this timestep
            d_h = dH[:, t, :] + d_h_next

            # derivative of activation
            d_activ = self.d_act(h_t) * d_h

            # gradients
            d_Wxh += X[:, t, :].T @ d_activ
            d_Whh += h_prev.T @ d_activ
            d_bh += d_activ.sum(axis=0, keepdims=True)

            # gradient wrt inputs
            d_X[:, t, :] = d_activ @ self.weights_xh.T

            # gradient wrt previous hidden state
            d_h_next = d_activ @ self.weights_hh.T

        # store gradients
        self.d_Wxh = d_Wxh
        self.d_Whh = d_Whh
        self.d_bh  = d_bh

        return d_X