# models.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

import datetime
import pickle
from . import losses
from .gradients import SGD
from .layers import MLP, DenseLayer, RNNLayer, GRULayer
from .processing import Scaling, train_test_split
import warnings
import matplotlib.pyplot as plt
from . import activation_functions as AF
from . import initiations as In
from . import backend as BE

VERSION = "0.2.3"
LIST_VERSIONS_COMPATIBLE = [VERSION]

def im2col(X, K, stride, padding):
    B, H, W, C = X.shape

    # Pad input
    X_padded = BE.xp.pad(X,
                      ((0,0), (padding,padding), (padding,padding), (0,0)),
                      mode='constant')


    H_p, W_p = X_padded.shape[1], X_padded.shape[2]

    H_out = (H_p - K) // stride + 1
    W_out = (W_p - K) // stride + 1
    
    shape = (B, H_out, W_out, K, K, C)
    strides = (
        X_padded.strides[0],
        stride * X_padded.strides[1],
        stride * X_padded.strides[2],
        X_padded.strides[1],
        X_padded.strides[2],
        X_padded.strides[3],
    )

    #cols = cols.reshape(B * H_out * W_out, K*K*C)
    #return cols, H_out, W_out

    X_strided = BE.xp.lib.stride_tricks.as_strided(
        X_padded, shape=shape, strides=strides
    )

    # Reshape to im2col matrix
    X_col = X_strided.reshape(B * H_out * W_out, K * K * C)
    return X_col, H_out, W_out

def col2im(dX_col, X_shape, K, stride=1, padding=0, H_out=None, W_out=None):
    B, H, W, C = X_shape

    # Initialize padded gradient
    dX_padded = BE.xp.zeros((B, H + 2*padding, W + 2*padding, C), dtype=dX_col.dtype)

    dX_strided = dX_col.reshape(B, H_out, W_out, K, K, C)

    for i in range(K):
        for j in range(K):
            dX_padded[:, 
                      i:i + H_out*stride:stride,
                      j:j + W_out*stride:stride,
                      :] += dX_strided[:, :, :, i, j, :]

    # Remove padding
    if padding > 0:
        return dX_padded[:, padding:-padding, padding:-padding, :]
    return dX_padded

def load_model_all(model_file):
    dummy_1 = type(CNN())
    dummy_2 = type(SequentialModel())
    try:
        loaded_model = pickle.load(open(model_file, 'rb'))
        if type(loaded_model) in (dummy_1, dummy_2):
            print("Model loaded")
        else:
            print("Object loaded (not model!):", type(loaded_model))
        return loaded_model
    except ImportError as e:
        raise ValueError("Couldn't load model:", e)    

def save_model_all(model, model_file_name = "model.pickle", weights_only = False, clear_cache = False):
    dummy_1 = type(CNN())
    dummy_2 = type(SequentialModel())
    try:
        pickle.dump(model, open(model_file_name, 'wb'))
        if type(model) in (dummy_1, dummy_2):
            print("Model saved as:", model_file_name)
        else:
            print(f"Object {type(model)} saved as:", model_file_name)
    except ImportError as e:
        raise ValueError("Couldn't save model:", e)

def load_CNN_model(path, flag_gpu):
    try:
        model = CNN(gpu=flag_gpu)
        model.load_model(path)
        return model
    except ImportError as e:
        raise ValueError("Couldn't load CNN model:", e)

def load_Sequential_model(path, flag_gpu):
    try:
        model = SequentialModel(gpu=flag_gpu)
        model.load_model(path)
        return model
    except ImportError as e:
        raise ValueError("Couldn't load Sequential model:", e)
    
def load_Recurrent_model(path, flag_gpu):
    try:
        model = RecurrentModel(gpu=flag_gpu)
        model.load_model(path)
        return model
    except ImportError as e:
        raise ValueError("Couldn't load Sequential model:", e)

def load_model(path):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
            version = model["version"]

            if version not in LIST_VERSIONS_COMPATIBLE:
                raise ValueError(f"Loaded model is of a version '{version}' - it is not compatible with current version {VERSION}. List of compatible versions {LIST_VERSIONS_COMPATIBLE}.")
            flag_gpu = model["gpu"]
            
            if model["model"] == "CNN":
                model_import = load_CNN_model(path, flag_gpu)
            elif model["model"] == "SequentialModel":
                model_import = load_Sequential_model(path, flag_gpu)
            elif model["model"] == "RecurrentModel":
                model_import = load_Recurrent_model(path, flag_gpu)
            else:
                raise KeyError(f"Couldn't recognize model '{model['model']}'.")
            return model_import
    except ImportError as e:
        raise ValueError("Couldn't load model:", e)

class SequentialModel:
    """
    Initialize the Sequential Model. The Neural Network follows a Multi-Layer Perceptron (MLP) architecture.

    **Manual Workflow**:
    1. Initialize the model:
        >>> model = SequentialModel()
    2. Define layer specifications (inputs, neurons, activation, initialization):
        >>> ly_info = [(X.shape[1], 10, "relu", "he"), (10, 1, 'linear', 'he')]
    3. Build the network:
        >>> model.construct(ly_info, learning_rate=0.01)
    4. Train the model:
        >>> model.fit(X_train, y_train, epochs=1000, batch_size=32, graphical=False, real_time=False, log_plot=False)
    5. Predict outputs:
        >>> y_pred = model.predict(X_test)

    **Automatic Workflow**:
    1. Initialize the model:
        >>> model = SequentialModel()
    2. Define layer specifications:
        >>> ly_info = [(X.shape[1], 10, "relu", "he"), (10, 1, 'linear', 'he')]
    3. Run the entire process automatically:
        >>> y_pred, final_pred_train, X_train, X_test, y_train, y_test = model.automatic(
            X=X, 
            y=y, 
            layers_info=ly_info, 
            learning_rate=0.01, 
            epochs=1000, 
            batch_size=32, 
            split_test_percentage=0.3, 
            scaling='minmax', 
            graphical=True, 
            real_time=False, 
            log_plot=True, 
            test_loss=True, 
            scatter_comparison=True
        )
    """

    def __init__(self, gpu = False):
        self.mlp = None
        self.optimizer = None
        self.learning_rate = None
        
        if gpu:
            if BE.gpu_available:
                print("Computing on GPU")
                self.flag_gpu = True
            else:
                print("GPU not available. Computing on CPU")
                self.processing_unit = "cpu"
                self.flag_gpu = False
        else:
            print("Computing on CPU")
            self.processing_unit = "cpu"
            self.flag_gpu = False
            BE.use_cpu()

    def construct(self, layers_info, learning_rate = 0.01, batch_norm = False, dropout = False):
        """
        Constructs the Dense Layers of the Neural Network while initializing the SGD optimizer.\n
        Parameters
        ----------
        :param layer_info: *list* - Information regarding how to construct the dense layers. Needs to be in the format of a list:\n[(#inputs, #neurons, 'activation function', 'initiation function'), ...].\n
        :param learning_rate: *float* - Hyperparameter that controls the step size.\n
        :param batch_norm: *bool* - Allows batch normalization.
        :param droput: *bool* - Allows dropout

        Example
        -------
            >>> model = SequentialModel()
        >>> ly_info = [(X.shape[1], 10, "relu", "he"), (10, 1, 'linear', 'he')]
        >>> model.construct(ly_info, learning_rate=0.2)
        """
        self.layers_info = layers_info
        self.mlp = MLP(layers_info, batch_norm=batch_norm, dropout=dropout)
        self.optimizer = SGD(learning_rate)
        self.learning_rate = learning_rate

        if layers_info[-1][2].lower() == "softmax":
            print("The last layer of the MLP has 'softmax' activation. The model will use 'cross-entropy' as the loss function.")

    def fit(self, X_train, y_train, epochs, batch_size, wd = 1e-4, loss_function = 'mse', graphical = False, real_time = False, log_plot = False):
        """
        Performs the train loop for each epoch.\n
        Parameters
        ----------
        :params X_train: *array* - X split for the training\n
        :params y_train: *array* - y split for the training\n
        :params epochs: *int* - Number of epochs\n
        :params batch_size: *int* - Size of each batch\n
        :params wd: *float* - Hyperparameter for the model regularization (weight decay)\n
        :params loss_function: *str* - Loss function to utilize during training ('mse', 'mae', 'cross-entropy', or 'huber' (or 'huber:delta' where delta is the hyperparameter. e.g. 'huber:1.3'))\n
            N.B.: for classification, the last layer should be "softmax" activated. This forces the loss function to be 'cross-entropy'.\n
        :params graphical: *bool* - Display the Loss graph at the end of the fitting\n
        :params real_time: *bool* - Display the Loss graph in real time\n
        :params log_plot: *bool* - Display the Loss graph in semilogy scale

        Returns
        -------
        :return final_pred_train: *array* - Final prediction during training

        Example
        -------
            >>> model = SequentialModel()
        >>> ly_info = [(X.shape[1], 10, "relu", "he"), (10, 1, 'linear', 'he')]
        >>> model.construct(ly_info, learning_rate=0.2)
        >>> final_pred = model.fit(X_train, y_train, epochs, batch_size, graphical = False, real_time = False, log_plot = False)
        """

        num_samples = X_train.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        try:
            tmp = loss_function.split(sep=':')
            self.delta = float(tmp[1])
            loss_function = tmp[0]
        except:
            loss_function = tmp[0]
            self.delta = 1

        if self.mlp.layers[-1].activation == "softmax":
            self.loss_func = losses.cross_entropy
            self.loss_gradient = losses.cross_entropy_der
        elif loss_function.lower() == "mse":
            self.loss_func = losses.MSE
            self.loss_gradient = losses.MSE_der
        elif loss_function.lower() == "mae":
            self.loss_func = losses.MAE
            self.loss_gradient = losses.MAE_der
        elif loss_function.lower() == "huber":
            self.loss_func = losses.Huber
            self.loss_gradient = losses.Huber_der
        elif loss_function.lower() == "cross-entropy":
            self.loss_func = losses.cross_entropy
            self.loss_gradient = losses.cross_entropy_der
        else:
            raise ValueError(f"Loss function '{loss_function}' not found. Please input: 'MSE', 'MAE', 'Cross-entropy' or 'Huber' (or 'Huber:delta' e.g. 'Huber:1.3').")


        if real_time == True and graphical == False:
            warnings.warn("The parameter graphical is set to False while real_time is True. Assuming graphical = True.")
            graphical = True

        if graphical:
            loss_list = []
        if real_time:
            plt.figure()
            plt.ion()

        print(f"Training for {epochs} Epochs with learning rate: {self.learning_rate:.2g}")


        for epoch in range(epochs):
            # Shuffle the data at the beginning of each epoch
            idx = BE.xp.random.permutation(num_samples)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]
            
            tot_loss = 0
            
            # Process data in mini-batches
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                
                y_pred = self.mlp.forward(X_batch)
                try:
                    loss = self.loss_func(y_true=y_batch, y_pred=y_pred)
                except:
                    loss = self.loss_func(y_true=y_batch, y_pred=y_pred, delta = self.delta)

                tot_loss += BE.xp.mean(loss)
                
                try:
                    d_loss_wrt_pred = self.loss_gradient(y_true=y_batch, y_pred=y_pred)
                except:
                    d_loss_wrt_pred = self.loss_gradient(y_true=y_batch, y_pred=y_pred, delta=self.delta)

                self.mlp.backward(d_loss_wrt_pred=d_loss_wrt_pred, wd=wd)
    
                for layer in self.mlp.layers:
                    self.optimizer.update(layer)
                
            # Average loss over all batches for reporting
            avg_loss = tot_loss / num_batches
            if (epoch + 1) % (epochs // 10 if epochs >=10 else 1) == 0 or epoch == 0:
                print(f"Epoch {epoch+1:4d}/{epochs}, Loss: {avg_loss:.5f}")
            if graphical:
                loss_list.append(avg_loss)
            if real_time:
                plt.clf()
                if log_plot: plt.semilogy(BE.to_numpy(BE.xp.arange(1, epoch+2, step = 1)), BE.to_numpy(BE.xp.array(loss_list)), linestyle = '-')
                else: plt.plot(BE.to_numpy(BE.xp.arange(1, epoch+2, step = 1)), BE.to_numpy(BE.xp.array(loss_list)), linestyle = '-')
                plt.title(f"Average loss ({num_batches} batches) over the epochs (current: {epoch})")
                plt.xlabel("Epoch")
                plt.ylabel("Average loss")
                plt.pause(5e-3)

                
        
        self.final_pred = self.mlp.forward(X_train)
        try:
            self.final_loss = self.loss_func(y_train, self.final_pred)
        except:
            self.final_loss = self.loss_func(y_train, self.final_pred, self.delta)
        if self.mlp.layers[-1].activation == "softmax": print(f"\nFinal 'cross-entropy' loss on training data: {BE.xp.mean(self.final_loss):.5f}")
        else: print(f"\nFinal '{loss_function}' loss on training data: {self.final_loss:.5f}")
        if graphical:
            plt.clf()
            if real_time:plt.ioff()
            if log_plot:plt.semilogy(BE.to_numpy(BE.xp.arange(1, epoch+2, step = 1)), BE.to_numpy(BE.xp.array(loss_list)), linestyle = '-')
            else: plt.plot(BE.to_numpy(BE.xp.arange(1, epochs+1, step = 1)), BE.to_numpy(BE.xp.array(loss_list)), linestyle = '-')
            plt.title(f"Average loss ({num_batches} batches) over the epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Average loss")
            plt.show()

        return BE.to_numpy(self.final_pred)
    
    def predict(self, X_test):
        """
        Make predictions using the testing features.\n
        Parameter
        ---------
        :param X_test: *array* - Data array used for the testing of the model.

        Return
        ------
        :return y_pred: *array* - Prediction array

        Example
        -------
            >>> model = SequentialModel()
        >>> ly_info = [(X_train.shape[1], 10, "relu", "he"), (10, 1, 'linear', 'he')]
        >>> model.construct(ly_info, learning_rate=0.2)
        >>> model.fit(X_train, y_train, epochs, batch_size, graphical = False, real_time = False, log_plot = False)
        >>> y_pred = model.predict(X_test)
        """
        y_pred = self.mlp.forward(X_test)
        return y_pred
    
    def automatic(self, X, y, layers_info, learning_rate = 0.01, epochs = 100, batch_size = 1, wd = 1e-4, loss_function = 'mse', split_test_percentage = 0.3, scaling = None, batch_norm = False, dropout = False, graphical = False, real_time = False, log_plot = False, test_loss = False, scatter_comparison = False):
        """
        Performs the *construction*, *fitting* and *prediction* based on the parameters given.\n
        Parameters
        ---------
        :param X: *array* - Features array.\n
        :param y: *array* - Target array.\n
        :param layer_info: *list* - Information regarding how to construct the dense layers. Needs to be in the format of a list:\n[(#inputs, #neurons, 'activation function', 'initiation function'), ...].\n
        :param learning_rate: *float* - Hyperparameter that controls the step size.\n
        :param epochs: *int* - Number of iterations for the training loop.\n
        :param batch_size: *int* - Size of the batches used in the training loop.\n
        :params wd: *float* - Hyperparameter for the model regularization (weight decay)\n
        :params loss_function: *str* - Loss function to utilize during training ('mse', 'mae', 'cross-entropy' or 'huber' (or 'huber:delta' where delta is the hyperparameter. e.g. 'huber:1.3'))\n
        :param split_test_percentage: *float* - Percentage of the total array size used to obtain the Test arrays.\n
        :param scaling: *str* - Name of the scaling function to utilize (can be None): 'zscore', 'minmax', 'log', or 'mean'.\n
        :param batch_norm: *bool* - Include batch normalization to the MLP architecture.\n
        :param dropout: *bool* - Include dropout to the MLP architecture.\n
        :param graphical: *bool* - Display the Loss graph at the end of the fitting.\n
        :param real_time: *bool* - Display the Loss graph in real time.\n
        :param log_plot: *bool* - Display the Loss graph in semilogy scale\n
        :param test_loss: *bool* - Prints the loss of the predicted and test data\n
        :param scatter_comparison: *bool* - Display the scatter plot Test vs. Predicted.\n

        Returns
        -------
        :return y_pred: *array* - The model's predictions during testing
        :return final_pred_train: *array* - Final prediction during training
        :return X_train: *array* - Input array used in the training
        :return X_test: *array* - Input array used in the testing
        :return y_train: *array* - Target array used in the training
        :return y_test: *array* - Target array used in the testing

        Example
        -------
            >>> model = SequentialModel()
        >>> ly_info = [(X.shape[1], 10, "relu", "he"), (10, 1, 'linear', 'he')]
        >>> y_pred, final_pred_train, X_train, X_test, y_train, y_test = model.automatic(X, y, layers_info, learning_rate = 0.01, epochs = 100, batch_size = 1, wd = 0.01, loss_function = 'mse', split_test_percentage = 0.3, scaling = None, graphical = False, real_time = False, log_plot = False, test_loss = False, scatter_comparison = False)
        """
        print("Starting the automatic pipeline.\n")
        if scaling == None:
            warnings.warn("No scaling method specified. Applying no scaling.\n")
        elif isinstance(scaling, str):
            scale = Scaling()
            tmp = X
            if scaling.lower() == "zscore":
                X = scale.zScore(X)
            elif scaling.lower() == "minmax":
                X = scale.MinMax(X)
            elif scaling.lower() == "log":
                X = scale.LogNorm(X)
            elif scaling.lower() == "mean":
                X = scale.MeanNorm(X)
            else:
                raise ValueError(f"Scaling method '{scaling}' not found. Please use 'zscore', 'minmax', 'log', or 'mean'.")
        else:
            raise ValueError(f"Type of scaling ({type(scaling)}) not allowed. Please input a string: 'zscore', 'minmax', 'log', or 'mean'.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, split_test_percentage)
        print(f"Split of the Train (len:{len(X_train)}) and Test (len:{len(X_test)}) completed.\n")

        if scaling != None and learning_rate >= BE.xp.std(tmp):
            sc_down = int(BE.xp.ceil(learning_rate/BE.xp.std(tmp) * 100))
            warnings.warn(f"Learning rate might be too high for the scaled data. It is recommended to reduce it by a factor x{sc_down}")

        self.construct(layers_info=layers_info, learning_rate=learning_rate, batch_norm=batch_norm, dropout=dropout)
        final_pred_train = self.fit(X_train=X_train, y_train=y_train, epochs=epochs, batch_size=batch_size, wd=wd, loss_function=loss_function, graphical=graphical, real_time=real_time, log_plot=log_plot)
        y_pred = self.predict(X_test=X_test)
        if test_loss:
            try:
                test_loss_value = self.loss_func(y_true=y_test, y_pred=y_pred)
            except:
                test_loss_value = self.loss_func(y_true=y_test, y_pred=y_pred, delta=self.delta)
            if self.loss_func != losses.cross_entropy: print(f"\n'{loss_function}' loss function result of Test vs. Predicted: {test_loss_value:2g}")
            else: print(f"\n'cross-entropy' loss function result of Test vs. Predicted: {BE.xp.mean(test_loss_value):2g}")
        if scatter_comparison:
            try:
                plt.scatter(x=BE.to_numpy(y_pred), y=BE.to_numpy(y_test))
                plt.title("Test data vs. Predicted data")
                plt.ylabel("Test")
                plt.xlabel("Prediction")
                if BE.xp.min(y_pred) < BE.xp.min(y_test):
                    limit_min = BE.xp.min(y_pred)
                else:
                    limit_min = BE.xp.min(y_test)
                if BE.xp.max(y_pred) > BE.xp.max(y_test):
                    limit_max = BE.xp.max(y_pred)
                else:
                    limit_max = BE.xp.max(y_test)
                plt.xlim(limit_min*0.85, limit_max + (limit_min*0.15))
                plt.ylim(limit_min*0.85, limit_max + (limit_min*0.15))
                plt.show()
            except:
                warnings.warn("Scatter comparison failed.")
            

        return y_pred, final_pred_train, X_train, X_test, y_train, y_test
    
    def save_weights(self, path):
        weights = {
            "mlp": [
                {
                    "W": layer.weights,
                    "b": layer.biases,
                }
                for layer in self.mlp.layers
            ]
        }
        if path == None:
            return weights
        else:
            with open(path, "wb") as f:
                pickle.dump(weights, f)
    
    def load_weights(self, weights):
        # Load MLP layers
        for layer, saved in zip(self.mlp.layers, weights["mlp"]):
            layer.weights = saved["W"]
            layer.biases = saved["b"]
    
    def save_model(self, path = "model.pickle"):
        self.batch_size = getattr(self, "batch_size", 16)
        try:
            weights = self.save_weights(path=None)
            model_architecture = {
                "version" : VERSION,
                "gpu": self.flag_gpu,
                "model" : "SequentialModel",
                "weights" : weights,
                "layers_info" : self.layers_info,
                "batch_size" : self.batch_size,
            }
            with open(path, "wb") as f:
                pickle.dump(model_architecture, f)
            print(f"Model saved: {path}")
        except ImportError as e:
            raise ValueError(f"{e}")
    
    def load_model(self, path):
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
                weights = model["weights"]
                layers_info = model["layers_info"]
                batch_size = model["batch_size"]
                flag_gpu = model["gpu"]
            
            self.batch_size = batch_size
            
            if self.flag_gpu != flag_gpu and self.flag_gpu == False:
                raise ValueError("Model loaded was trained on GPU, but the GPU is currently not available. In future version it'll be compatible!")

            self.construct(layers_info=layers_info)
            self.load_weights(weights=weights)
            print("Sequential model loaded")
        except ImportError as e:
            raise ValueError(e)



class BatchNorm2D:
    def __init__(self, num_channels, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum

        self.gamma = BE.xp.ones((1, 1, 1, num_channels))
        self.beta  = BE.xp.zeros((1, 1, 1, num_channels))

        self.running_mean = BE.xp.zeros((1, 1, 1, num_channels))
        self.running_var  = BE.xp.ones((1, 1, 1, num_channels))

    def forward(self, X, training=True):
        if training:
            # mean/var over batch + spatial dims
            self.mu = BE.xp.mean(X, axis=(0,1,2), keepdims=True)
            self.var = BE.xp.var(X, axis=(0,1,2), keepdims=True)

            self.x_norm = (X - self.mu) / BE.xp.sqrt(self.var + self.eps)
            out = self.gamma * self.x_norm + self.beta

            # update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mu
            self.running_var  = self.momentum * self.running_var  + (1 - self.momentum) * self.var
        else:
            x_norm = (X - self.running_mean) / BE.xp.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta

        self.X = X
        return out
            
    def backward(self, d_out):
        B, H, W, C = d_out.shape
        N = B * H * W  # number of elements per channel

        # gradients for gamma and beta
        self.d_gamma = BE.xp.sum(d_out * self.x_norm, axis=(0,1,2), keepdims=True)
        self.d_beta  = BE.xp.sum(d_out, axis=(0,1,2), keepdims=True)

        dx_norm = d_out * self.gamma

        dvar = BE.xp.sum(dx_norm * (self.X - self.mu) * -0.5 * (self.var + self.eps)**(-3/2),
                    axis=(0,1,2), keepdims=True)

        dmu = (BE.xp.sum(dx_norm * -1 / BE.xp.sqrt(self.var + self.eps),
                    axis=(0,1,2), keepdims=True)
            + dvar * BE.xp.mean(-2 * (self.X - self.mu), axis=(0,1,2), keepdims=True))

        dX = (dx_norm / BE.xp.sqrt(self.var + self.eps)
            + dvar * 2 * (self.X - self.mu) / N
            + dmu / N)
                
        return dX
            
    def update(self, lr):
        self.gamma -= lr * self.d_gamma
        self.beta  -= lr * self.d_beta

class CNN:
    """
    Initialize the Convolution model.\n
    This model combines convolutional layers, max-pooling, batch normalization, and dropout, followed by a fully connected Multi-Layer Perceptron (MLP) classifier
    
    Parameters
    ----------
    :param filter_size: *int* - Size of the (square) filter used in the convolution layers. Default is 3
    :param num_filters: *int* - Number of filters used in the convolution layers. Default is 4
    :param padding: *int* - Amount of padding applied to the array for each convolution layer. Default is 1
    :param stride: *int* - Amount of strading applied to the array for each convolution layer. Default is 1
    :param activation_function: *str* - Activation function used for the convolution layers. Default is 'relu'
    :param init_function: *str* - Initiation function used for the weights of each convolution layer. Default is 'he'
    :param num_channels: *int* - Number of color channels in the images uploaded. Default is 3
    :param pool_size: *int* - Size of the max-pooling applied. Default is 2
   
    Example
    ----------

    **1. Initialize the model:**
        >>> model_cnn = CNN(filter_size = 3, num_filters = 16, padding = 1, stride = 1, activation_function = "relu", init_function = "he")


    **2. Gather the size of the input for the layer specification:**

        >>> input_size = model_cnn.get_input_size(X_train=X_train)

    **3. Define layer specifications (inputs, neurons, activation, initialization):**
        
        >>> layer_info = [
            (input_size, 256, 'relu', 'he'),
            (256, 128, 'relu', 'he'),
            (128, y_train.shape[1], 'softmax', 'he')
        ]
    
    Be aware that for classification models, the last layer needs to follow the rule (n_input, y_train.shape[1], 'softmax', 'he')
    
    **4. Build the network:**

        >>> model_cnn.construct(layers_info=layer_info, learning_rate=1e-4)

    **5. Train the model:**

        >>> final_pred = model_cnn.fit(X_train = X_train, y_train=y_train, epochs = 250, batch_size = 32, wd = 0, graphical = True, real_time = False, log_plot = False)
    
    **6. Predict outputs:**

        >>> y_pred = model_cnn.predict(X_test)
    """

    def __init__(self, filter_size = 3, num_filters = 4, padding = 1, stride = 1, num_channels = 3, activation_function = "relu", init_function = "he", pool_size = 2, gpu = True):
        self.mlp = None
        self.optimizer = None
        self.batchnorm = None
        self.dropout = None
        self.learning_rate = None
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.activation_function = activation_function
        self.init_function = init_function
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.og_params_conv = [self.filter_size, self.num_filters, self.padding, self.stride]
        self.est_time = None

        params_int = [filter_size, num_filters, padding, stride, num_channels, pool_size]
        params_label = ["filter_size"," num_filters", "padding", "stride", "num_channels", "pool_size"]

        if gpu:
            if BE.gpu_available:
                print("Computing on GPU")
                self.flag_gpu = True
            else:
                print("GPU not available. Computing on CPU")
                self.processing_unit = "cpu"
                self.flag_gpu = False
        else:
            print("Computing on CPU")
            self.processing_unit = "cpu"
            self.flag_gpu = False
            BE.use_cpu()

        for p in range(len(params_int)):
            if not isinstance(params_int[p], int):
                raise ValueError(f"Parameter '{params_label[p]}' of the CNN class must be an integer.")
        
        try:
            if activation_function.lower() not in ['sigmoid', 'relu', 'tanh', 'linear']:
                raise ValueError(f"Activation function for convolution '{activation_function}' not found.\nPlease choose: 'sigmoid', 'relu', 'tanh', or 'linear'.")
        except:
            raise ValueError(f"Activation function for convolution '{activation_function}' not found.\nPlease choose: 'sigmoid', 'relu', 'tanh', or 'linear'.")

        """if self.init_function == "random":
            self.weights = BE.xp.random.rand(self.num_filters, self.filter_size, self.filter_size, num_channels) * 0.01
        elif self.init_function in ("xavier", "glorot"):
            stddev = BE.xp.sqrt(2/(self.num_filters + num_channels))
            self.weights = BE.xp.random.randn(self.num_filters, self.filter_size, self.filter_size, num_channels) * stddev
        elif self.init_function in ("he", "kaiming"):
            limit = BE.xp.sqrt(6/self.num_filters)
            self.weights = BE.xp.random.uniform(-limit, limit, (self.num_filters, self.filter_size, self.filter_size, num_channels))
        else:
            raise ValueError(f"Initialization function '{self.init_function}' not found.\nPlease choose: 'random', 'xavier' or 'he'.")
            
        self.biases = BE.xp.random.rand(self.num_filters) * 0"""

        self.conv1a = self.ConvolutionLayer(filter_size = filter_size, num_filters = 1*num_filters, padding = padding, stride = stride, num_channels = num_channels, activation_function = activation_function, init_function = init_function, pool_size = pool_size)
        self.conv2a = self.ConvolutionLayer(filter_size = filter_size, num_filters = 2*num_filters, padding = padding, stride = stride, num_channels = 1*num_filters, activation_function = activation_function, init_function = init_function, pool_size = pool_size)
        self.conv3a = self.ConvolutionLayer(filter_size = filter_size, num_filters = 4*num_filters, padding = padding, stride = stride, num_channels = 2*num_filters, activation_function = activation_function, init_function = init_function, pool_size = pool_size)

        self.conv1b = self.ConvolutionLayer(filter_size = filter_size, num_filters = 1*num_filters, padding = padding, stride = stride, num_channels = 1*num_filters, activation_function = activation_function, init_function = init_function, pool_size = pool_size)
        self.conv2b = self.ConvolutionLayer(filter_size = filter_size, num_filters = 2*num_filters, padding = padding, stride = stride, num_channels = 2*num_filters, activation_function = activation_function, init_function = init_function, pool_size = pool_size)
        self.conv3b = self.ConvolutionLayer(filter_size = filter_size, num_filters = 4*num_filters, padding = padding, stride = stride, num_channels = 4*num_filters, activation_function = activation_function, init_function = init_function, pool_size = pool_size)

    def construct(self, layers_info, learning_rate = 0.01, batch_norm = True, dropout = True):
        """
        Constructs the Dense Layers of the Neural Network while initializing the SGD optimizer.\n
        Parameters
        ----------
        :param layer_info: Information regarding how to construct the dense layers. Needs to be in the format of a list:\n[(#inputs, #neurons, 'activation function', 'initiation function'), ...].\n
        :param learning_rate: Hyperparameter that controls the step size.\n
        :param batch_norm: *bool* - Include batch normalization to the MLP architecture.\n
        :param dropout: *bool* - Include dropout to the MLP architecture.\n

        Example
        -------
            >>> model_cnn = CNN(filter_size = 3, num_filters = 16, padding = 1, stride = 1, activation_function = "relu", init_function = "he")
        >>> input_size = model_cnn.get_input_size(X_train=X_train)
        >>> layer_info = [
            (input_size, 256, 'relu', 'he'),
            (256, 128, 'relu', 'he'),
            (128, y_train.shape[1], 'softmax', 'he')
        ]
        >>> model_cnn.construct(layers_info=layer_info, learning_rate=1e-4)
        """
        self.layers_info = layers_info
        self.mlp = MLP(layers_info, batch_norm=batch_norm, dropout=dropout)
        self.optimizer = SGD(learning_rate)
        self.learning_rate = learning_rate

        if layers_info[-1][2].lower() != "softmax":
            warnings.warn(
                "The last layer of the CNN's MLP head does not use 'softmax'. "
                "Cross-entropy loss expects probability distributions, so training may be unstable "
                "or fail to converge. It is strongly recommended to use a softmax output for "
                "multi-class classification.",
                UserWarning)     

    class ConvolutionLayer:

        def __init__(self, filter_size = 3, num_filters = 4, padding = 1, stride = 1, num_channels = 3, activation_function = "relu", init_function = "he", pool_size = 2):
            self.mlp = None
            self.optimizer = None
            self.learning_rate = None
            self.filter_size = filter_size
            self.stride = stride
            self.padding = padding
            self.activation_function = activation_function
            self.init_function = init_function
            self.num_filters = num_filters
            self.og_params_conv = [self.filter_size, self.num_filters, self.padding, self.stride]

            self.bn = BatchNorm2D(num_channels=num_filters)

            if self.init_function == "random":
                self.weights = BE.xp.random.rand(self.num_filters, self.filter_size, self.filter_size, num_channels) * 0.01
            elif self.init_function in ("xavier", "glorot"):
                stddev = BE.xp.sqrt(2/(self.num_filters + num_channels))
                self.weights = BE.xp.random.randn(self.num_filters, self.filter_size, self.filter_size, num_channels) * stddev
            elif self.init_function in ("he", "kaiming"):
                fan_in = self.filter_size * self.filter_size * num_channels
                limit = BE.xp.sqrt(6.0 / fan_in)
                self.weights = BE.xp.random.uniform(-limit, limit,
                                                (self.num_filters, self.filter_size,
                                                self.filter_size, num_channels))
            else:
                raise ValueError(f"Initialization function '{self.init_function}' not found.\nPlease choose: 'random', 'xavier' or 'he'.")
                
            self.biases = BE.xp.random.rand(self.num_filters) * 0

        def forward(self, X):
            batch_size, im_h, im_w, num_channels = X.shape

            if im_w <= im_h: min_size = im_w
            else: min_size = im_h
            if self.filter_size > min_size: raise ValueError(f"Filter size ({self.filter_size}) too large. Please reduce it below {min_size}")

            layer_conv = self.conv_forward_im2col(
                X = X,
                W = self.weights,
                b = self.biases,
                stride = self.stride,
                padding = self.padding
            )

            layer_conv = self.bn.forward(layer_conv, training=True)

            if self.activation_function == "sigmoid":
                self.layer_conv = AF.sigmoid(layer_conv)
            elif self.activation_function == "relu":
                self.layer_conv = AF.reLU(layer_conv)
            elif self.activation_function == "linear":
                self.layer_conv = AF.linear(layer_conv)
            elif self.activation_function == "tanh":
                self.layer_conv = AF.tanh(layer_conv)
            else:
                raise ValueError(f"Activation function for convolution'{self.activation_function}' not found.\nPlease choose: 'sigmoid', 'relu', 'tanh', or 'linear'.")
            
            return self.layer_conv

        def conv_forward_im2col(self, X, W, b, stride, padding):
            K = W.shape[1]
            X_col, H_out, W_out = im2col(X, K, stride, padding)
            W_col = W.reshape(self.num_filters, -1)

            self.cache = (X, X_col, W, W_col, stride, padding, H_out, W_out)

            out = X_col @ W_col.T + b
            out = out.reshape(X.shape[0], H_out, W_out, self.num_filters)

            return out
        
        def backward(self, d_out):

            # 1. Activation derivative
            if self.activation_function == "relu":
                d_act = AF.reLU_der(self.layer_conv) * d_out
            elif self.activation_function == "sigmoid":
                d_act = AF.sigmoid_der(self.layer_conv) * d_out
            elif self.activation_function == "tanh":
                d_act = AF.tanh_der(self.layer_conv) * d_out
            else:
                d_act = d_out  # linear

            d_out = self.bn.backward(d_act)

            dX, dW, db = self.conv_backward_im2col(d_out)

            self.d_weights = dW
            self.d_biases = db

            return dX
        
        def conv_backward_im2col(self, d_out):
            X, X_col, W, W_col, stride, padding, H_out, W_out = self.cache
            B = X.shape[0]
            K = W.shape[1]

            d_out_col = d_out.reshape(B*H_out*W_out, self.num_filters)

            # Gradients
            dW_col = d_out_col.T @ X_col
            db = BE.xp.sum(d_out_col, axis=0)

            dW = dW_col.reshape(W.shape)

            dX_col = d_out_col @ W_col

            dX = col2im(dX_col, X.shape, K, stride, padding, H_out=H_out, W_out=W_out)
            
            return dX, dW, db
        
        def maxpool_forward(self, X, pool_size, stride):
            B, H, W, C = X.shape

            H_out = (H - pool_size) // stride + 1
            W_out = (W - pool_size) // stride + 1

            out = BE.xp.zeros((B, H_out, W_out, C))
            self.maxpool_cache = (X, pool_size, stride)

            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    w_start = w * stride

                    window = X[:, 
                            h_start:h_start+pool_size,
                            w_start:w_start+pool_size,
                            :]

                    out[:, h, w, :] = BE.xp.max(window, axis=(1, 2))

            return out
            
        def maxpool_backward(self, d_out):
            X, pool_size, stride = self.maxpool_cache
            B, H, W, C = X.shape

            H_out = d_out.shape[1]
            W_out = d_out.shape[2]

            dX = BE.xp.zeros_like(X)

            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride
                    w_start = w * stride

                    window = X[:, 
                            h_start:h_start+pool_size,
                            w_start:w_start+pool_size,
                            :]

                    max_vals = BE.xp.max(window, axis=(1, 2), keepdims=True)
                    mask = (window == max_vals)

                    dX[:, 
                    h_start:h_start+pool_size,
                    w_start:w_start+pool_size,
                    :] += mask * d_out[:, h:h+1, w:w+1, :]

            return dX
        
        def update(self, lr, wd):
            self.bn.update(lr)
            self.weights -= lr * (self.d_weights + wd * self.weights)
            self.biases  -= lr * self.d_biases
            self.weights = self.weights.astype(BE.xp.float32)
            self.biases = self.biases.astype(BE.xp.float32)
            self.d_weights = BE.xp.zeros_like(self.weights)
            self.d_biases  = BE.xp.zeros_like(self.biases)
    
    def ConvolutionBlock(self, X):
        X = self.conv1a.forward(X)
        X = self.conv1b.forward(X)
        X = self.conv1b.maxpool_forward(X, 2, 2)

        X = self.conv2a.forward(X)
        X = self.conv2b.forward(X)
        X = self.conv2b.maxpool_forward(X, 2, 2)

        X = self.conv3a.forward(X)
        X = self.conv3b.forward(X)
        X = self.conv3b.maxpool_forward(X, 2, 2)

        return X
    
    def ConvolutionBackward(self, d):
        # Backprop through Conv3 block
        d = self.conv3b.maxpool_backward(d)
        d = self.conv3b.backward(d)
        self.conv3b.update(self.learning_rate, self.wd)
        d = self.conv3a.backward(d)
        self.conv3a.update(self.learning_rate, self.wd)

        # Backprop through Conv2 block
        d = self.conv2b.maxpool_backward(d)
        d = self.conv2b.backward(d)
        self.conv2b.update(self.learning_rate, self.wd)
        d = self.conv2a.backward(d)
        self.conv2a.update(self.learning_rate, self.wd)

        # Backprop through Conv1 block
        d = self.conv1b.maxpool_backward(d)
        d = self.conv1b.backward(d)
        self.conv1b.update(self.learning_rate, self.wd)
        d = self.conv1a.backward(d)
        self.conv1a.update(self.learning_rate, self.wd)


    def fit(self, X_train, y_train, epochs, batch_size, wd = 1e-4, graphical = False, real_time = False, log_plot = False, report = False):
        """
        Performs the train loop for each epoch.\n
        Parameters
        ----------
        :params X_train: X split for the training\n
        :params y_train: y split for the training\n
        :params epochs: Number of epochs\n
        :params batch_size: Size of each batch\n
        :params batch_size: Size of each batch\n
        :params wd: Hyperparameter for the model regularization (weight decay)\n
        :params graphical: Display the Loss graph at the end of the fitting\n
        :params real_time: Display the Loss graph in real time\n
        :params log_plot: Display the Loss graph in semilogy scale

        N.B.: for classification, the last layer should be "softmax" activated. This forces the loss function to be 'cross-entropy'.\n


        Returns
        -------
        :return final_pred_train: Final prediction during training

        Example
        -------
            >>> model_cnn = CNN(filter_size = 3, num_filters = 16, padding = 1, stride = 1, activation_function = "relu", init_function = "he")
        >>> input_size = model_cnn.get_input_size(X_train=X_train)
        >>> layer_info = [
            (input_size, 256, 'relu', 'he'),
            (256, 128, 'relu', 'he'),
            (128, y_train.shape[1], 'softmax', 'he')
        ]
        >>> model_cnn.construct(layers_info=layer_info, learning_rate=1e-4)
        >>> final_pred = model_cnn.fit(X_train = X_train, y_train=y_train, epochs = 250, batch_size = 32, wd = 0, graphical = True, real_time = False, log_plot = False)
        """

        self.wd = wd
        num_samples = len(X_train)
        num_batches = (num_samples + batch_size - 1) // batch_size
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_classes = y_train.shape[1]

        if self.est_time != None:
            self.est_time *= num_samples * (epochs +1) * 2 * 1.5
        
            if self.est_time > 3600:
                print(f"Estimated time: {int(BE.xp.floor(self.est_time/3600))} hours and {int((self.est_time/3600 - BE.xp.floor(self.est_time/3600))*60)} minutes.\n")
            else:
                print(f"Estimated time: {int(BE.xp.ceil(self.est_time/60))} minutes.\n")
        
        self.loss_func = losses.cross_entropy
        self.loss_gradient = losses.cross_entropy_der

        if real_time == True and graphical == False:
            warnings.warn("The parameter graphical is set to False while real_time is True. Assuming graphical = True.")
            graphical = True

        if graphical:
            loss_list = []
        if real_time:
            plt.figure()
            plt.ion()

        print(f"Training for {epochs} Epochs with learning rate: {self.learning_rate:.2g}")

        for epoch in range(epochs):
            # Shuffle the data at the beginning of each epoch
            idx = BE.xp.random.permutation(num_samples)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]
            
            tot_loss = 0
            
            # Process data in mini-batches
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                batch_conv_outputs_flat = []
                
                conv_out = self.ConvolutionBlock(X=X_batch)

                B = conv_out.shape[0]
                batch_conv_outputs_flat = conv_out.reshape(B, -1)

                y_pred = self.mlp.forward(batch_conv_outputs_flat)
                
                try:
                    loss = self.loss_func(y_true=y_batch, y_pred=y_pred)
                except:
                    loss = self.loss_func(y_true=y_batch, y_pred=y_pred, delta = self.delta)

                tot_loss += loss
                
                try:
                    d_loss_wrt_pred = self.loss_gradient(y_true=y_batch, y_pred=y_pred)
                except:
                    d_loss_wrt_pred = self.loss_gradient(y_true=y_batch, y_pred=y_pred, delta=self.delta)

                d_mlp = self.mlp.backward(d_loss_wrt_pred=d_loss_wrt_pred, wd=wd)

                d_conv_input = d_mlp.reshape(conv_out.shape[0], *conv_out.shape[1:])

                self.ConvolutionBackward(d=d_conv_input)

                for layer in self.mlp.layers:
                    self.optimizer.update(layer)                
                
            # Average loss over all batches for reporting
            avg_loss = tot_loss / num_batches
            if (epoch + 1) % (epochs // 10 if epochs >=10 else 1) == 0 or epoch == 0:
                print(f"Epoch {epoch+1:4d}/{epochs}, Loss: {avg_loss:.5f}")
            if graphical:
                loss_list.append(avg_loss)
            if real_time:
                plt.clf()
                if log_plot: plt.semilogy(BE.to_numpy(BE.xp.arange(1, epoch+2, step = 1)), BE.to_numpy(BE.xp.array(loss_list)), linestyle = '-')
                else: plt.plot(BE.to_numpy(BE.xp.arange(1, epoch+2, step = 1)), BE.to_numpy(BE.xp.array(loss_list)), linestyle = '-')
                plt.title(f"Average loss ({num_batches} batches) over the epochs (current: {epoch})")
                plt.xlabel("Epoch")
                plt.ylabel("Average loss")
                plt.pause(5e-3)
        
        self.final_loss = 0
        y_final = []

        for i in range(0, num_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            batch_conv_outputs_flat = []
            
            conv_out = self.ConvolutionBlock(X=X_batch)

            B = conv_out.shape[0]
            batch_conv_outputs_flat = conv_out.reshape(B, -1)

            self.final_pred = self.mlp.forward(batch_conv_outputs_flat)

            try:
                self.final_loss += self.loss_func(y_batch, self.final_pred)
            except:
                self.final_loss += self.loss_func(y_batch, self.final_pred, self.delta)
            
            y_final.append(self.final_pred)

        self.final_pred = BE.xp.concatenate(y_final, axis=0)
        self.final_loss /= num_batches

        print(f"\nFinal 'cross-entropy' loss on training data: {BE.xp.mean(self.final_loss):.5f}")

        if graphical:
            plt.clf()
            if real_time:plt.ioff()
            if log_plot:plt.semilogy(BE.to_numpy(BE.xp.arange(1, epoch+2, step = 1)), BE.to_numpy(BE.xp.array(loss_list)), linestyle = '-')
            else: plt.plot(BE.to_numpy(BE.xp.arange(1, epochs+1, step = 1)), BE.to_numpy(BE.xp.array(loss_list)), linestyle = '-')
            plt.title(f"Average loss ({num_batches} batches) over the epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Average loss")
            plt.show()
        
        if report:
            from .metrics import Metrics
            print("\n****Report on final prediction****")
            metrics_report = Metrics(y_test=y_train, y_pred=self.final_pred)
            metrics_report.report(graphical=graphical)
            print()

        return self.final_pred
    
    def predict(self, X_test, batch_size = None):
        """
        Make predictions using the testing features.\n
        Parameter
        ---------
        :param X_test: Data array used for the testing of the model.

        Return
        ------
        :return y_pred: Prediction array

        Example
        -------
            >>> model_cnn = CNN(filter_size = 3, num_filters = 16, padding = 1, stride = 1, activation_function = "relu", init_function = "he")
        >>> input_size = model_cnn.get_input_size(X_train=X_train)
        >>> layer_info = [
            (input_size, 256, 'relu', 'he'),
            (256, 128, 'relu', 'he'),
            (128, y_train.shape[1], 'softmax', 'he')
        ]
        >>> model_cnn.construct(layers_info=layer_info, learning_rate=1e-4)
        >>> final_pred = model_cnn.fit(X_train = X_train, y_train=y_train, epochs = 250, batch_size = 32, wd = 0, graphical = True, real_time = False, log_plot = False)
        >>> y_pred = model_cnn.predict(X_test=X_train)
        """
        y_pred_final = []
        num_samples = len(X_test)

        if batch_size == None:
            try:
                batch_size = self.batch_size
            except:
                batch_size = num_samples//10

        for i in range(0, num_samples, batch_size):
            X_batch = X_test[i:i+batch_size]
            batch_conv_outputs_flat = []
            conv_out = self.ConvolutionBlock(X=X_batch)

            B = conv_out.shape[0]
            batch_conv_outputs_flat = conv_out.reshape(B, -1)

            y_pred = self.mlp.forward(batch_conv_outputs_flat)
            y_pred_final.append(y_pred)

        y_pred = BE.xp.concatenate(y_pred_final, axis=0)
        return y_pred
  
    def get_input_size(self, X_train):
        """
        Gathers the size of the input of the first layer.\n
        Parameter
        ---------
        :param X_train: *ndarray* Data array used for the training of the model.

        Return
        ------
        :return in_size: Size of the input array

        Example
        -------
            >>> model_cnn = CNN(filter_size = 3, num_filters = 16, padding = 1, stride = 1, activation_function = "relu", init_function = "he")
        >>> input_size = model_cnn.get_input_size(X_train=X_train)
        >>> layer_info = [
            (input_size, 256, 'relu', 'he'),
            (256, 128, 'relu', 'he'),
            (128, y_train.shape[1], 'softmax', 'he')
        ]
        """

        in_time = datetime.datetime.now()

        batch_conv_outputs_flat = []
        
        conv_out = self.ConvolutionBlock(X=X_train[0:1])
        for out in conv_out:    
            batch_conv_outputs_flat.append(out.flatten())
        batch_conv_outputs_flat = BE.xp.array(batch_conv_outputs_flat)
    
        in_size = batch_conv_outputs_flat.size

        fin_time = datetime.datetime.now()

        if BE.gpu_available == True:
            self.est_time = None
        else:
            self.est_time = (fin_time - in_time).total_seconds()

        return in_size
    
    def save_weights(self, path):
        weights = {
            "conv1a": {
                "W": self.conv1a.weights,
                "b": self.conv1a.biases,
                "gamma": self.conv1a.bn.gamma,
                "beta": self.conv1a.bn.beta,
                "running_mean": self.conv1a.bn.running_mean,
                "running_var": self.conv1a.bn.running_var
            },
            "conv1b": {
                "W": self.conv1b.weights,
                "b": self.conv1b.biases,
                "gamma": self.conv1b.bn.gamma,
                "beta": self.conv1b.bn.beta,
                "running_mean": self.conv1b.bn.running_mean,
                "running_var": self.conv1b.bn.running_var
            },
            "conv2a": {
                "W": self.conv2a.weights,
                "b": self.conv2a.biases,
                "gamma": self.conv2a.bn.gamma,
                "beta": self.conv2a.bn.beta,
                "running_mean": self.conv2a.bn.running_mean,
                "running_var": self.conv2a.bn.running_var
            },
            "conv2b": {
                "W": self.conv2b.weights,
                "b": self.conv2b.biases,
                "gamma": self.conv2b.bn.gamma,
                "beta": self.conv2b.bn.beta,
                "running_mean": self.conv2b.bn.running_mean,
                "running_var": self.conv2b.bn.running_var
            },
            "conv3a": {
                "W": self.conv3a.weights,
                "b": self.conv3a.biases,
                "gamma": self.conv3a.bn.gamma,
                "beta": self.conv3a.bn.beta,
                "running_mean": self.conv3a.bn.running_mean,
                "running_var": self.conv3a.bn.running_var
            },
            "conv3b": {
                "W": self.conv3b.weights,
                "b": self.conv3b.biases,
                "gamma": self.conv3b.bn.gamma,
                "beta": self.conv3b.bn.beta,
                "running_mean": self.conv3b.bn.running_mean,
                "running_var": self.conv3b.bn.running_var
            },
            "mlp": [
                {
                    "W": layer.weights,
                    "b": layer.biases
                }
                for layer in self.mlp.layers
            ]
        }
        if path == None:
            return weights
        else:
            with open(path, "wb") as f:
                pickle.dump(weights, f)

    def load_weights(self, path):
        with open(path, "rb") as f:
            weights = pickle.load(f)

        # Load conv layers
        self.conv1a.weights = weights["conv1a"]["W"]
        self.conv1a.biases = weights["conv1a"]["b"]
        self.conv1a.bn.gamma = weights["conv1a"]["gamma"]
        self.conv1a.bn.beta = weights["conv1a"]["beta"]
        self.conv1a.bn.running_mean = weights["conv1a"]["running_mean"]
        self.conv1a.bn.running_var = weights["conv1a"]["running_var"]

        self.conv1b.weights = weights["conv1b"]["W"]
        self.conv1b.biases = weights["conv1b"]["b"]
        self.conv1b.bn.gamma = weights["conv1b"]["gamma"]
        self.conv1b.bn.beta = weights["conv1b"]["beta"]
        self.conv1b.bn.running_mean = weights["conv1b"]["running_mean"]
        self.conv1b.bn.running_var = weights["conv1b"]["running_var"]


        self.conv2a.weights = weights["conv2a"]["W"]
        self.conv2a.biases = weights["conv2a"]["b"]
        self.conv2a.bn.gamma = weights["conv2a"]["gamma"]
        self.conv2a.bn.beta = weights["conv2a"]["beta"]
        self.conv2a.bn.running_mean = weights["conv2a"]["running_mean"]
        self.conv2a.bn.running_var = weights["conv2a"]["running_var"]

        self.conv2b.weights = weights["conv2b"]["W"]
        self.conv2b.biases = weights["conv2b"]["b"]
        self.conv2b.bn.gamma = weights["conv2b"]["gamma"]
        self.conv2b.bn.beta = weights["conv2b"]["beta"]
        self.conv2b.bn.running_mean = weights["conv2b"]["running_mean"]
        self.conv2b.bn.running_var = weights["conv2b"]["running_var"]


        self.conv3a.weights = weights["conv3a"]["W"]
        self.conv3a.biases = weights["conv3a"]["b"]
        self.conv3a.bn.gamma = weights["conv3a"]["gamma"]
        self.conv3a.bn.beta = weights["conv3a"]["beta"]
        self.conv3a.bn.running_mean = weights["conv3a"]["running_mean"]
        self.conv3a.bn.running_var = weights["conv3a"]["running_var"]

        self.conv3b.weights = weights["conv3b"]["W"]
        self.conv3b.biases = weights["conv3b"]["b"]
        self.conv3b.bn.gamma = weights["conv3b"]["gamma"]
        self.conv3b.bn.beta = weights["conv3b"]["beta"]
        self.conv3b.bn.running_mean = weights["conv3b"]["running_mean"]
        self.conv3b.bn.running_var = weights["conv3b"]["running_var"]

        # Load MLP layers
        for layer, saved in zip(self.mlp.layers, weights["mlp"]):
            layer.weights = saved["W"]
            layer.biases = saved["b"]

    def load_weights_internal(self, weights):

        # Load conv layers
        self.conv1a.weights = weights["conv1a"]["W"]
        self.conv1a.biases = weights["conv1a"]["b"]
        self.conv1a.bn.gamma = weights["conv1a"]["gamma"]
        self.conv1a.bn.beta = weights["conv1a"]["beta"]
        self.conv1a.bn.running_mean = weights["conv1a"]["running_mean"]
        self.conv1a.bn.running_var = weights["conv1a"]["running_var"]

        self.conv1b.weights = weights["conv1b"]["W"]
        self.conv1b.biases = weights["conv1b"]["b"]
        self.conv1b.bn.gamma = weights["conv1b"]["gamma"]
        self.conv1b.bn.beta = weights["conv1b"]["beta"]
        self.conv1b.bn.running_mean = weights["conv1b"]["running_mean"]
        self.conv1b.bn.running_var = weights["conv1b"]["running_var"]


        self.conv2a.weights = weights["conv2a"]["W"]
        self.conv2a.biases = weights["conv2a"]["b"]
        self.conv2a.bn.gamma = weights["conv2a"]["gamma"]
        self.conv2a.bn.beta = weights["conv2a"]["beta"]
        self.conv2a.bn.running_mean = weights["conv2a"]["running_mean"]
        self.conv2a.bn.running_var = weights["conv2a"]["running_var"]

        self.conv2b.weights = weights["conv2b"]["W"]
        self.conv2b.biases = weights["conv2b"]["b"]
        self.conv2b.bn.gamma = weights["conv2b"]["gamma"]
        self.conv2b.bn.beta = weights["conv2b"]["beta"]
        self.conv2b.bn.running_mean = weights["conv2b"]["running_mean"]
        self.conv2b.bn.running_var = weights["conv2b"]["running_var"]


        self.conv3a.weights = weights["conv3a"]["W"]
        self.conv3a.biases = weights["conv3a"]["b"]
        self.conv3a.bn.gamma = weights["conv3a"]["gamma"]
        self.conv3a.bn.beta = weights["conv3a"]["beta"]
        self.conv3a.bn.running_mean = weights["conv3a"]["running_mean"]
        self.conv3a.bn.running_var = weights["conv3a"]["running_var"]

        self.conv3b.weights = weights["conv3b"]["W"]
        self.conv3b.biases = weights["conv3b"]["b"]
        self.conv3b.bn.gamma = weights["conv3b"]["gamma"]
        self.conv3b.bn.beta = weights["conv3b"]["beta"]
        self.conv3b.bn.running_mean = weights["conv3b"]["running_mean"]
        self.conv3b.bn.running_var = weights["conv3b"]["running_var"]

        # Load MLP layers
        for layer, saved in zip(self.mlp.layers, weights["mlp"]):
            layer.weights = saved["W"]
            layer.biases = saved["b"]

    def save_model(self, path = "model_CNN.pickle"):
        self.batch_size = getattr(self, "batch_size", 16)
        self.num_channels = getattr(self, "num_channels", 3)
        try:
            weights = self.save_weights(path=None)
            model_architecture = {
                "version" : VERSION,
                "gpu": self.flag_gpu,
                "model" : "CNN",
                "weights" : weights,
                "layers_info" : self.layers_info,
                "batch_size" : self.batch_size,
                "filter_size": self.filter_size,
                "stride" : self.stride,
                "padding" : self.padding,
                "activation_function" : self.activation_function,
                "init_function" : self.init_function,
                "num_filters" : self.num_filters,
                "num_channels" : self.num_channels
            }
            with open(path, "wb") as f:
                pickle.dump(model_architecture, f)
            print(f"Model saved: {path}")
        except ImportError as e:
            raise ValueError(f"{e}")
    
    def load_model(self, path):
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
                weights = model["weights"]
                layers_info = model["layers_info"]
                batch_size = model["batch_size"]
                filter_size = model["filter_size"]
                stride = model["stride"]
                padding = model["padding"]
                activation_function = model["activation_function"]
                init_function = model["init_function"]
                num_filters = model["num_filters"]
                num_channels = model["num_channels"]
                flag_gpu = model["gpu"]

            self.conv1a = self.ConvolutionLayer(filter_size = filter_size, num_filters = 1*num_filters, padding = padding, stride = stride, num_channels = num_channels, activation_function = activation_function, init_function = init_function, pool_size = 2)
            self.conv2a = self.ConvolutionLayer(filter_size = filter_size, num_filters = 2*num_filters, padding = padding, stride = stride, num_channels = 1*num_filters, activation_function = activation_function, init_function = init_function, pool_size = 2)
            self.conv3a = self.ConvolutionLayer(filter_size = filter_size, num_filters = 4*num_filters, padding = padding, stride = stride, num_channels = 2*num_filters, activation_function = activation_function, init_function = init_function, pool_size = 2)

            self.conv1b = self.ConvolutionLayer(filter_size = filter_size, num_filters = 1*num_filters, padding = padding, stride = stride, num_channels = 1*num_filters, activation_function = activation_function, init_function = init_function, pool_size = 2)
            self.conv2b = self.ConvolutionLayer(filter_size = filter_size, num_filters = 2*num_filters, padding = padding, stride = stride, num_channels = 2*num_filters, activation_function = activation_function, init_function = init_function, pool_size = 2)
            self.conv3b = self.ConvolutionLayer(filter_size = filter_size, num_filters = 4*num_filters, padding = padding, stride = stride, num_channels = 4*num_filters, activation_function = activation_function, init_function = init_function, pool_size = 2)

            self.batch_size = batch_size

            if self.flag_gpu != flag_gpu and self.flag_gpu == False:
                raise ValueError("Model loaded was trained on GPU, but the GPU is currently not available. In future version it'll be compatible!")

            self.construct(layers_info=layers_info)
            self.load_weights_internal(weights=weights)
            print("CNN model loaded")
        except ImportError as e:
            raise ValueError(e)

class RecurrentModel:
    def __init__(self, rnn_type = 'gru', gpu=False):
        self.layers = []
        self.learning_rate = None
        self.optimizer = None
        self.rnn_type = rnn_type.lower()
        
        if gpu:
            if BE.gpu_available:
                print("Computing on GPU")
                self.flag_gpu = True
            else:
                print("GPU not available. Computing on CPU")
                self.processing_unit = "cpu"
                self.flag_gpu = False
        else:
            print("Computing on CPU")
            self.processing_unit = "cpu"
            self.flag_gpu = False
            BE.use_cpu()

    def construct(self, input_dim, hidden_dim, output_dim, activation_function, init_function = "he", learning_rate=0.001, random_scale = 0.01, act_function_rnn='tanh', init_function_rnn='xavier', random_scale_rnn=0.01, many_to_one=True, batch_norm = False, dropout = False):
        """
        Construct the RNN and Dense layers\n
        Parameters
        ----------
        :params input_dim: *int* - Input dimensions
        :params hidden_dim: *int* - Hidden dimensions
        :params output_dim: *int* - Output dimensions
        :param activation_function: *str* - Activation function for the Dense layer
        :param init_function: *str* - Initiation function for the Dense layer
        Example
        -------
            >>> text = "hellohellohellohellohello"
        >>> X, y, c2i, i2c, vocab = generate_text_dataset(text, seq_len=5) #external function
        >>> model = RecurrentModel(gpu=False)
        >>> model.construct(
                input_dim=vocab,
                hidden_dim=32,
                output_dim=vocab,
                activation_function="softmax",
                act_function_rnn="tanh",
                many_to_one=True
            )
        """
        self.construction = [input_dim, hidden_dim, output_dim, activation_function, init_function, learning_rate, random_scale, act_function_rnn, init_function_rnn, random_scale_rnn, many_to_one, batch_norm, dropout]
        self.learning_rate = learning_rate
        if self.rnn_type == 'gru':
            self.rnn = GRULayer(input_dim=input_dim, hidden_dim=hidden_dim, init_function=init_function_rnn)  
        else:
            self.rnn = RNNLayer(input_dim=input_dim, hidden_dim=hidden_dim, activation_function=act_function_rnn, init_function=init_function_rnn, random_scale=random_scale_rnn)
        self.dense = DenseLayer(extras= [batch_norm, dropout], num_inputs=hidden_dim, num_neurons=output_dim, activation_function=activation_function, init_function=init_function, random_scale=random_scale)
        self.many_to_one = many_to_one
        self.optimizer = SGD(learning_rate=learning_rate)

    def forward(self, X):
        H, cache = self.rnn.forward(X)
        if self.many_to_one:
            out = self.dense.forward(H[:, -1, :])
        else:
            out = self.dense.forward(H.reshape(-1, H.shape[-1]))
        return out, cache

    def backward(self, dOut, cache, wd = 1e-5):
        d_H = self.dense.backward(dOut, wd=wd)
        if self.many_to_one:
            d_H_full = BE.xp.zeros((d_H.shape[0], cache["X"].shape[1], d_H.shape[1]))
            d_H_full[:, -1, :] = d_H
            d_X = self.rnn.backward(d_H_full, cache)
        else:
            d_H_seq = d_H.reshape(cache["X"].shape[0], cache["X"].shape[1], -1)
            d_X = self.rnn.backward(d_H_seq, cache)
        return d_X

    def update(self, wd):
        # SGD update
        self.rnn.update(learning_rate=self.learning_rate, wd=wd)
        self.optimizer.update(layer=self.dense)

    def fit(self, X_train, y_train, epochs, batch_size=16, wd = 1e-4, loss_function = 'mse', graphical = False, real_time = False, log_plot = False):
        """
        Performs the train loop for each epoch.\n
        Parameters
        ----------
        :params X_train: X split for the training\n
        :params y_train: y split for the training\n
        :params epochs: Number of epochs\n
        :params batch_size: Size of each batch\n
        :params batch_size: Size of each batch\n
        :params wd: Hyperparameter for the model regularization (weight decay)\n
        :params loss_function: Loss function to utilize during training ('MSE', 'MAE' or 'Huber' (or 'Huber:delta' where delta is the hyperparameter. e.g. 'Huber:1.3'))\n
            N.B.: for classification, the last layer should be "softmax" activated. This forces the loss function to be 'cross-entropy'.\n
        :params graphical: Display the Loss graph at the end of the fitting\n
        :params real_time: Display the Loss graph in real time\n
        :params log_plot: Display the Loss graph in semilogy scale

        Returns
        -------
        :return final_pred_train: Final prediction during training

        Example
        -------
            >>> text = "hellohellohellohellohello"
        >>> X, y, c2i, i2c, vocab = generate_text_dataset(text, seq_len=5) #external function
        >>> model = RecurrentModel(gpu=False)
        >>> model.construct(
                input_dim=vocab,
                hidden_dim=32,
                output_dim=vocab,
                activation_function="softmax",
                act_function_rnn="tanh",
                many_to_one=True
            )
        >>> model.fit(X, y, epochs=100, batch_size=16, loss_function="cross-entropy", graphical=True)
        """

        num_samples = X_train.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        try:
            tmp = loss_function.split(sep=':')
            self.delta = float(tmp[1])
            loss_function = tmp[0]
        except:
            loss_function = tmp[0]
            self.delta = 1

        if self.dense.activation == "softmax":
            self.loss_func = losses.cross_entropy
            self.loss_gradient = losses.cross_entropy_der
            if loss_function.lower() != "cross-entropy":
                warnings.warn("For 'softmax' activation, the 'cross-entropy loss function is required. Switch applied automatically.", UserWarning)
        elif loss_function.lower() == "mse":
            self.loss_func = losses.MSE
            self.loss_gradient = losses.MSE_der
        elif loss_function.lower() == "mae":
            self.loss_func = losses.MAE
            self.loss_gradient = losses.MAE_der
        elif loss_function.lower() == "huber":
            self.loss_func = losses.Huber
            self.loss_gradient = losses.Huber_der
        elif loss_function.lower() == "cross-entropy":
            self.loss_func = losses.cross_entropy
            self.loss_gradient = losses.cross_entropy_der
        else:
            raise ValueError(f"Loss function '{loss_function}' not found. Please input: 'MSE', 'MAE' or 'Huber' (or 'Huber:delta' e.g. 'Huber:1.3').")


        if real_time == True and graphical == False:
            warnings.warn("The parameter graphical is set to False while real_time is True. Assuming graphical = True.")
            graphical = True

        if graphical:
            loss_list = []
        if real_time:
            plt.figure()
            plt.ion()

        print(f"Training for {epochs} Epochs with learning rate: {self.learning_rate:.2g}")


        for epoch in range(epochs):
            # Shuffle the data at the beginning of each epoch
            idx = BE.xp.random.permutation(num_samples)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]
            
            tot_loss = 0
            
            # Process data in mini-batches
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                
                y_pred, cache = self.forward(X_batch)
                try:
                    loss = self.loss_func(y_true=y_batch, y_pred=y_pred)
                except:
                    loss = self.loss_func(y_true=y_batch, y_pred=y_pred, delta = self.delta)

                tot_loss += BE.xp.mean(loss)
                
                try:
                    d_loss_wrt_pred = self.loss_gradient(y_true=y_batch, y_pred=y_pred)
                except:
                    d_loss_wrt_pred = self.loss_gradient(y_true=y_batch, y_pred=y_pred, delta=self.delta)

                self.backward(dOut = d_loss_wrt_pred, cache=cache, wd = wd)
    
                self.update(wd = wd)
                
            # Average loss over all batches for reporting
            avg_loss = tot_loss / num_batches
            if (epoch + 1) % (epochs // 10 if epochs >=10 else 1) == 0 or epoch == 0:
                print(f"Epoch {epoch+1:4d}/{epochs}, Loss: {avg_loss:.5f}")
            if graphical:
                loss_list.append(avg_loss)
            if real_time:
                plt.clf()
                if log_plot: plt.semilogy(BE.to_numpy(BE.xp.arange(1, epoch+2, step = 1)), BE.to_numpy(BE.xp.array(loss_list)), linestyle = '-')
                else: plt.plot(BE.to_numpy(BE.xp.arange(1, epoch+2, step = 1)), BE.to_numpy(BE.xp.array(loss_list)), linestyle = '-')
                plt.title(f"Average loss ({num_batches} batches) over the epochs (current: {epoch})")
                plt.xlabel("Epoch")
                plt.ylabel("Average loss")
                plt.pause(5e-3)

        self.final_pred, cache = self.forward(X_train)
        try:
            self.final_loss = self.loss_func(y_train, self.final_pred)
        except:
            self.final_loss = self.loss_func(y_train, self.final_pred, self.delta)
        if self.dense.activation == "softmax": print(f"\nFinal 'cross-entropy' loss on training data: {BE.xp.mean(self.final_loss):.5f}")
        else: print(f"\nFinal '{loss_function}' loss on training data: {self.final_loss:.5f}")
        if graphical:
            plt.clf()
            if real_time:plt.ioff()
            if log_plot:plt.semilogy(BE.to_numpy(BE.xp.arange(1, epoch+2, step = 1)), BE.to_numpy(BE.xp.array(loss_list)), linestyle = '-')
            else: plt.plot(BE.to_numpy(BE.xp.arange(1, epochs+1, step = 1)), BE.to_numpy(BE.xp.array(loss_list)), linestyle = '-')
            plt.title(f"Average loss ({num_batches} batches) over the epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Average loss")
            plt.show()

        return BE.to_numpy(self.final_pred)
    
    def predict(self, X_test):
        """
        Make predictions using the testing features.\n
        Parameter
        ---------
        :param X_test: Data array used for the testing of the model.

        Return
        ------
        :return y_pred: Prediction array

        Example
        -------
            >>> text = "hellohellohellohellohello"
        >>> X, y, c2i, i2c, vocab = generate_text_dataset(text, seq_len=5) #external function
        >>> model = RecurrentModel(gpu=False)
        >>> model.construct(
                input_dim=vocab,
                hidden_dim=32,
                output_dim=vocab,
                activation_function="softmax",
                act_function_rnn="tanh",
                many_to_one=True
            )
        >>> model.fit(X, y, epochs=100, batch_size=16, loss_function="cross-entropy", graphical=True)
        >>> test_text = "hello"
        >>> X_test, y_test, c2i, i2c, vocab = generate_text_dataset(test_text, seq_len=4)
        >>> y_pred = model.predict(X_test)
        """
        y_pred, cache = self.forward(X_test)
        return y_pred
    
    def save_weights(self, path):
        if self.rnn_type == "gru":
            weights = {
                "dense":
                    {
                        "W": self.dense.weights,
                        "b": self.dense.biases,
                    },
                "rnn":
                    {
                        "Wz": self.rnn.weights_W_z,
                        "Uz": self.rnn.weights_U_z,
                        "bz": self.rnn.biases_z,
                        "Wr": self.rnn.weights_W_r,
                        "Ur": self.rnn.weights_U_r,
                        "br": self.rnn.biases_r,
                        "Wh": self.rnn.weights_W_h,
                        "Uh": self.rnn.weights_U_h,
                        "bh": self.rnn.biases_h
                    }
            }
        else:
            weights = {
                "dense":
                    {
                        "W": self.dense.weights,
                        "b": self.dense.biases,
                    },
                "rnn":
                    {
                        "Wx": self.rnn.weights_xh,
                        "Wh": self.rnn.weights_hh,
                        "bh": self.rnn.biases_h
                    }
            }

        if path == None:
            return weights
        else:
            with open(path, "wb") as f:
                pickle.dump(weights, f)
    
    def load_weights(self, weights):
        # Load Dense layer
        self.dense.weights = weights['dense']['W']
        self.dense.biases = weights['dense']['b']

        # Load RNN layer
        if self.rnn_type == "gru":
            # Update
            self.rnn.weights_W_z = weights['rnn']['Wz']
            self.rnn.weights_U_z = weights['rnn']['Uz']
            self.rnn.biases_z = weights['rnn']['bz']
            # Reset
            self.rnn.weights_W_r = weights['rnn']['Wr']
            self.rnn.weights_U_r = weights['rnn']['Ur']
            self.rnn.biases_r = weights['rnn']['br']
            # Hidden state
            self.rnn.weights_W_h = weights['rnn']['Wh']
            self.rnn.weights_U_h = weights['rnn']['Uh']
            self.rnn.biases_h = weights['rnn']['bh']
        else:
            self.rnn.weights_xh = weights['rnn']['Wx']
            self.rnn.weights_hh = weights['rnn']['Wh']
            self.rnn.biases_h = weights['rnn']['bh']

    
    def save_model(self, path = "RNN_model.pickle"):
        self.batch_size = getattr(self, "batch_size", 16)
        try:
            weights = self.save_weights(path=None)
            model_architecture = {
                "version" : VERSION,
                "gpu": self.flag_gpu,
                "model" : "RecurrentModel",
                "type" : self.rnn_type,
                "weights" : weights,
                "batch_size" : self.batch_size,
                "construction": self.construction
            }
            with open(path, "wb") as f:
                pickle.dump(model_architecture, f)
            print(f"Model saved: {path}")
        except ImportError as e:
            raise ValueError(f"{e}")
    
    def load_model(self, path):
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
                type_model = model["type"]
                weights = model["weights"]
                batch_size = model["batch_size"]
                construction = model["construction"]
                flag_gpu = model["gpu"]
            
            self.batch_size = batch_size
            self.rnn_type = type_model
            
            if self.flag_gpu != flag_gpu and self.flag_gpu == False:
                raise ValueError("Model loaded was trained on GPU, but the GPU is currently not available. In future version it'll be compatible!")

            self.construct(*construction)
            self.load_weights(weights=weights)
            print("Recurrent model loaded")
        except ImportError as e:
            raise ValueError(e)
