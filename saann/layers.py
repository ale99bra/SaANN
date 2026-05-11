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

    def backward(self, d_H, cache):
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
            d_h = d_H[:, t, :] + d_h_next

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
    
    def update(self, learning_rate, wd=1e-5):
        self.weights_xh -= learning_rate * (self.d_Wxh + wd * self.weights_xh)
        self.weights_hh -= learning_rate * (self.d_Whh + wd * self.weights_hh)
        self.biases_h  -= learning_rate * self.d_bh

    
class GRULayer:

    def __init__(self, input_dim, hidden_dim, init_function = "xavier"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        if init_function == "random":
            warnings.warn(message="The 'random' initiation is unstable. 'xavier' initiation is recommended for GRU.", category=UserWarning)
            self.init_W = In.random_scaled_init
        elif init_function in ("xavier", "glorot"):
            self.init_W = In.xavier_init
        elif init_function in ("he", "kaiming"):
            warnings.warn(message="The 'he' initiation tends to explode. 'xavier' initiation is recommended for GRU.", category=UserWarning)
            self.init_W = In.he_init
        else:
            raise ValueError(f"Initialization function '{init_function}' not found.\nPlease choose: 'random', 'xavier' (recommended) or 'he'.")
        
        # Update gate block
        self.weights_W_z = self.init_W(input_dim, hidden_dim) 
        self.weights_U_z = self.init_W(hidden_dim, hidden_dim)
        self.biases_z = BE.xp.zeros((1, hidden_dim))
        self.d_Wz = BE.xp.zeros_like(self.weights_W_z)

        # Reset gate block
        self.weights_W_r = self.init_W(input_dim, hidden_dim) 
        self.weights_U_r = self.init_W(hidden_dim, hidden_dim)
        self.biases_r = BE.xp.zeros((1, hidden_dim))
        self.d_Wr = BE.xp.zeros_like(self.weights_W_r)

        # Candidate hidden state block
        self.weights_W_h = self.init_W(input_dim, hidden_dim) 
        self.weights_U_h = self.init_W(hidden_dim, hidden_dim)
        self.biases_h = BE.xp.zeros((1, hidden_dim))
        self.d_Wh = BE.xp.zeros_like(self.weights_W_h)
    
    def forward(self, X, h0 = None):
        """
        Perform the forward pass of the GRU layer\n

        Parameters
        ----------
        :param X: *array* - Data array of shape (batch, seq_len, input_dim)
        :param h0: *array* - Starting point - can be None

        Returns
        -------
        :param H: *array* - Hidden state array of shape (batch, seq_len, hidden_dim)
        :cache: *dict* - Dictionary for the backend
        """

        batch_size, seq_len, _ = X.shape

        if h0 is None:
            h_t = BE.xp.zeros((batch_size, self.hidden_dim))
            h0 = BE.xp.zeros_like(h_t)
        else:
            h_t = h0

        H = BE.xp.zeros((batch_size, seq_len, self.hidden_dim))

        z_list, r_list, h_tilde_list, h_list = [], [], [], []

        for t in range(seq_len):
            x_t = X[:, t, :]

            z_t = AF.sigmoid(BE.xp.dot(x_t, self.weights_W_z) + BE.xp.dot(h_t, self.weights_U_z) + self.biases_z)
            r_t = AF.sigmoid(BE.xp.dot(x_t, self.weights_W_r) + BE.xp.dot(h_t, self.weights_U_r) + self.biases_r)

            h_tilde = AF.tanh(BE.xp.dot(x_t, self.weights_W_h) + BE.xp.dot(r_t * h_t, self.weights_U_h) + self.biases_h)

            h_t = (1 - z_t) * h_tilde + z_t * h_t

            H[:, t, :] = h_t

            z_list.append(z_t)
            r_list.append(r_t)
            h_tilde_list.append(h_tilde)
            h_list.append(h_t)

        cache = {
            "X": X,
            "z": z_list,
            "r": r_list,
            "h_tilde": h_tilde_list,
            "h": h_list,
            "h0": h0
        }

        return H, cache        

    def backward(self, d_H, cache):
        X = cache["X"]
        batch, seq_len, _ = X.shape

        # Update gate block
        self.d_Wz = BE.xp.zeros_like(self.weights_W_z)
        self.d_Uz = BE.xp.zeros_like(self.weights_U_z)
        self.d_bz = BE.xp.zeros_like(self.biases_z)
        z_list = cache["z"]

        # Reset gate block
        self.d_Wr = BE.xp.zeros_like(self.weights_W_r)
        self.d_Ur = BE.xp.zeros_like(self.weights_U_r)
        self.d_br = BE.xp.zeros_like(self.biases_r)
        r_list = cache["r"]

        # Hidden state block
        self.d_Wh = BE.xp.zeros_like(self.weights_W_h)
        self.d_Uh = BE.xp.zeros_like(self.weights_U_h)
        self.d_bh = BE.xp.zeros_like(self.biases_h)
        h_tilde_list = cache["h_tilde"]

        h_list = cache["h"]
        h0 = cache["h0"]

        d_X = BE.xp.zeros_like(X)
        d_h_next = BE.xp.zeros((batch, self.hidden_dim))

        for t in reversed(range(seq_len)):
            x_t = X[:, t, :]
            h_t = h_list[t]
            h_prev = h_list[t-1] if t > 0 else h0
            z_t = z_list[t]
            r_t = r_list[t]
            h_tilde = h_tilde_list[t]

            # total gradient at this timestep
            d_h = d_H[:, t, :] + d_h_next

            # gradients h_prev
            d_h_tilde = d_h * (1 - z_t)
            d_z = d_h * (h_prev - h_tilde)
            d_h_prev = d_h * z_t

            # gradient of h_tilde
            d_h_tilde_raw = AF.tanh_der(h_tilde) * d_h_tilde

            # gradient of z
            d_z_raw = AF.sigmoid_der(z_t) * d_z

            # gradient of z
            d_r = BE.xp.dot(d_h_tilde_raw, self.weights_U_h.T) * h_prev
            d_r_raw = d_r * r_t * (1 - r_t)

            # gradient wrt parameters
            # Update gate
            self.d_Wz += BE.xp.dot(x_t.T, d_z_raw)
            self.d_Uz += BE.xp.dot(h_prev.T, d_z_raw)
            self.d_bz += BE.xp.sum(d_z_raw, axis=0, keepdims=True)

            # Reset gate
            self.d_Wr += BE.xp.dot(x_t.T, d_r_raw)
            self.d_Ur += BE.xp.dot(h_prev.T, d_r_raw)
            self.d_br += BE.xp.sum(d_r_raw, axis=0, keepdims=True)

            # Candidate hidden state
            self.d_Wh += BE.xp.dot(x_t.T, d_h_tilde_raw)
            self.d_Uh += BE.xp.dot((r_t * h_prev).T, d_h_tilde_raw)
            self.d_bh += BE.xp.sum(d_h_tilde_raw, axis=0, keepdims=True)

            # gradients wrt inputs
            d_X[:, t, :] = (
                BE.xp.dot(d_z_raw, self.weights_W_z.T) +
                BE.xp.dot(d_r_raw, self.weights_W_r.T) +
                BE.xp.dot(d_h_tilde_raw, self.weights_W_h.T)
            )

            # gradient wrt previous hidden state
            d_h_prev += (
                BE.xp.dot(d_z_raw, self.weights_U_z.T) + 
                BE.xp.dot(d_r_raw, self.weights_U_r.T) + 
                BE.xp.dot(d_h_tilde_raw, self.weights_U_h.T) * r_t 
            )

            d_h_next = d_h_prev

        return d_X
    
    def update(self, learning_rate, wd = 1e-5):
        for W, d_W in [
            (self.weights_W_z, self.d_Wz),
            (self.weights_U_z, self.d_Uz),
            (self.weights_W_r, self.d_Wr),
            (self.weights_U_r, self.d_Ur),
            (self.weights_W_h, self.d_Wh),
            (self.weights_U_h, self.d_Uh),
        ]:
            W -= learning_rate * (d_W + wd * W)

        self.biases_z -= learning_rate * self.d_bz
        self.biases_r -= learning_rate * self.d_br
        self.biases_h -= learning_rate * self.d_bh
