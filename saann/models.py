# models.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

import numpy as np
from .losses import MSE, MSE_der
from .gradients import SGD
from .layers import MLP
from .processing import Scaling, train_test_split
import warnings
import matplotlib.pyplot as plt

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

    def __init__(self):
        self.mlp = None
        self.optiomizer = None
        self.learning_rate = None

    def construct(self, layers_info, learning_rate = 0.01):
        """
        Constructs the Dense Layers of the Neural Network while initializing the SGD optimizer.\n
        Parameters
        ----------
        :param layer_info: Information regarding how to construct the dense layers. Needs to be in the format of a list:\n[(#inputs, #neurons, 'activation function', 'initiation function'), ...].\n
        :param learning_rate: Hyperparameter that controls the step size.\n

        Example
        -------
            >>> model = SequentialModel()
        >>> ly_info = [(X.shape[1], 10, "relu", "he"), (10, 1, 'linear', 'he')]
        >>> model.construct(ly_info, learning_rate=0.2)
        """
        self.mlp = MLP(layers_info)
        self.optimizer = SGD(learning_rate)
        self.learning_rate = learning_rate

    def fit(self, X_train, y_train, epochs, batch_size, graphical = False, real_time = False, log_plot = False):
        """
        Performs the train loop for each epoch.\n
        Parameters
        ----------
        :params epochs: number of epochs\n
        :params batch_size: Size of each batch\n
        :params X_train: X split for the training\n
        :params y_train: y split for the training\n
        :params batch_size: size of each batch\n
        :params graphical: Display the Loss graph at the end of the fitting\n
        :params real_time: Display the Loss graph in real time\n
        :params log_plot: Display the Loss graph in semilogy scale

        Returns
        -------
        :return final_pred_train: Final prediction during training

        Example
        -------
            >>> model = SequentialModel()
        >>> ly_info = [(X.shape[1], 10, "relu", "he"), (10, 1, 'linear', 'he')]
        >>> model.construct(ly_info, learning_rate=0.2)
        >>> final_pred = model.fit(X_train, y_train, epochs, batch_size, graphical = False, real_time = False, log_plot = False)
        """

        num_samples = X_train.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

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
            idx = np.random.permutation(num_samples)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]
            
            tot_loss = 0
            
            # Process data in mini-batches
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                
                y_pred = self.mlp.forward(X_batch)
                loss = MSE(y_batch, y_pred)
                tot_loss += loss

                d_loss_wrt_pred = MSE_der(y_true=y_batch, y_pred=y_pred)
                self.mlp.backward(d_loss_wrt_pred)
    
                for layer in self.mlp.layers:
                    self.optimizer.update(layer)
                
            # Average loss over all batches for reporting
            avg_loss = tot_loss / num_batches
            if (epoch + 1) % (epochs // 10 if epochs >=10 else 1) == 0 or epoch == 0:
                print(f"Epoch {epoch+1:4d}/{epochs}, Loss (MSE): {avg_loss:.5f}")
            if graphical:
                loss_list.append(avg_loss)
            if real_time:
                plt.clf()
                if log_plot: plt.semilogy(np.arange(1, epoch+2, step = 1), loss_list, linestyle = '-')
                else: plt.plot(np.arange(1, epoch+2, step = 1), loss_list, linestyle = '-')
                plt.title(f"Average loss (over the {num_batches} batches) over the epochs (current: {epoch})")
                plt.xlabel("Epoch")
                plt.ylabel("Average loss")
                plt.pause(5e-3)

                
        
        self.final_pred = self.mlp.forward(X_train)
        self.final_loss = MSE(y_train, self.final_pred)
        print(f"\nFinal MSE Loss on training data: {self.final_loss:.5f}")

        if graphical:
            plt.clf()
            if real_time:plt.ioff()
            if log_plot: plt.semilogy(np.arange(1, epoch+2, step = 1), loss_list, linestyle = '-')
            else: plt.plot(np.arange(1, epochs+1, step = 1), loss_list, linestyle = '-')
            plt.title(f"Average loss (over the {num_batches} batches) over the epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Average loss")
            plt.show()

        return self.final_pred
    
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
            >>> model = SequentialModel()
        >>> ly_info = [(X_train.shape[1], 10, "relu", "he"), (10, 1, 'linear', 'he')]
        >>> model.construct(ly_info, learning_rate=0.2)
        >>> model.fit(X_train, y_train, epochs, batch_size, graphical = False, real_time = False, log_plot = False)
        >>> y_pred = model.predict(X_test)
        """
        y_pred = self.mlp.forward(X_test)
        return y_pred
    
    def automatic(self, X, y, layers_info, learning_rate = 0.01, epochs = 100, batch_size = 1, split_test_percentage = 0.3, scaling = None, graphical = False, real_time = False, log_plot = False, test_loss = False, scatter_comparison = False):
        """
        Performs the *construction*, *fitting* and *prediction* based on the parameters given.\n
        Parameters
        ---------
        :param X: Features array.\n
        :param y: Target array.\n
        :param layer_info: Information regarding how to construct the dense layers. Needs to be in the format of a list:\n[(#inputs, #neurons, 'activation function', 'initiation function'), ...].\n
        :param learning_rate: Hyperparameter that controls the step size.\n
        :param epochs: Number of iterations for the training loop.\n
        :param batch_size: Size of the batches used in the training loop.\n
        :param split_test_percentage: Percentage of the total array size used to obtain the Test arrays.\n
        :param scaling: Name of the scaling function to utilize (can be None): 'zscore', 'minmax', 'log', or 'mean'.\n
        :param graphical: Display the Loss graph at the end of the fitting.\n
        :param real_time: Display the Loss graph in real time.\n
        :param log_plot: Display the Loss graph in semilogy scale\n
        :param test_loss: Returns and prints the loss of the predicted and test data\n
        :param scatter_comparison: Display the scatter plot Test vs. Predicted.\n

        Returns
        -------
        :return y_pred: The model's predictions during testing
        :return final_pred_train: Final prediction during training
        :return X_train: Input array used in the training
        :return X_test: Input array used in the testing
        :return y_train: Target array used in the training
        :return y_test: Target array used in the testing

        Example
        -------
            >>> model = SequentialModel()
        >>> ly_info = [(X.shape[1], 10, "relu", "he"), (10, 1, 'linear', 'he')]
        >>> y_pred, final_pred_train, X_train, X_test, y_train, y_test = model.automatic(X=X, y=y, layers_info=ly_info, learning_rate=0.01, epochs=1000, batch_size=32, split_test_percentage=0.3, scaling='minmax', graphical=True, real_time=False, log_plot = True, test_loss = True, scatter_comparison = True)
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

        if scaling != None and learning_rate >= np.std(tmp):
            sc_down = int(np.ceil(learning_rate/np.std(tmp) * 100))
            warnings.warn(f"Learning rate might be too high for the scaled data. It is recommended to reduce it by a factor x{sc_down}")

        self.construct(layers_info=layers_info, learning_rate=learning_rate)
        final_pred_train = self.fit(X_train=X_train, y_train=y_train, epochs=epochs, batch_size=batch_size, graphical=graphical, real_time=real_time, log_plot=log_plot)
        y_pred = self.predict(X_test=X_test)
        if test_loss:
            test_loss_value = MSE(y_true=y_test, y_pred=y_pred)
            print(f"\nMean Squared Error of Test vs. Predicted: {test_loss_value:2g}")
         
        if scatter_comparison:
            plt.scatter(x=y_pred, y=y_test)
            plt.title("Test data vs. Predicted data")
            plt.ylabel("Test")
            plt.xlabel("Prediction")
            if np.min(y_pred) < np.min(y_test):
                limit_min = np.min(y_pred)
            else:
                limit_min = np.min(y_test)
            if np.max(y_pred) > np.max(y_test):
                limit_max = np.max(y_pred)
            else:
                limit_max = np.max(y_test)
            plt.xlim(limit_min*0.85, limit_max + (limit_min*0.15))
            plt.ylim(limit_min*0.85, limit_max + (limit_min*0.15))
            plt.show()

        return y_pred, final_pred_train, X_train, X_test, y_train, y_test