import unittest
import numpy as np
from saann.models import SequentialModel, CNN, RecurrentModel
from saann.processing import train_test_split


class TestSequentialModel(unittest.TestCase):
    """Test suite for SequentialModel class"""

    def setUp(self):
        """Initialize test fixtures before each test"""
        self.model = SequentialModel()
        np.random.seed(42)
        self.X_train = np.random.randn(50, 5)
        self.y_train = np.random.randn(50, 1)
        self.X_test = np.random.randn(20, 5)

    def test_model_initialization(self):
        """Test that model initializes correctly"""
        self.assertIsNotNone(self.model)

    def test_construct_creates_layers(self):
        """Test that construct method creates layers"""
        layers_info = [(5, 10, "relu", "he"), (10, 1, "linear", "he")]
        self.model.construct(layers_info, learning_rate=0.01)

    def test_construct_sets_learning_rate(self):
        """Test that construct sets learning rate"""
        layers_info = [(5, 10, "relu", "he"), (10, 1, "linear", "he")]
        learning_rate = 0.001
        self.model.construct(layers_info, learning_rate=learning_rate)
        
        self.assertEqual(self.model.learning_rate, learning_rate)

    def test_fit_trains_without_error(self):
        """Test that fit method trains the model without errors"""
        layers_info = [(5, 10, "relu", "he"), (10, 1, "linear", "he")]
        self.model.construct(layers_info, learning_rate=0.01)
        
        # Should not raise any exception
        try:
            self.model.fit(
                self.X_train, 
                self.y_train, 
                epochs=2, 
                batch_size=32,
                loss_function='Huber',
                graphical=False,
                log_plot=False
            )
        except Exception as e:
            self.fail(f"fit() raised {type(e).__name__} unexpectedly: {e}")

    def test_predict_returns_correct_shape(self):
        """Test that predict returns correct output shape"""
        layers_info = [(5, 10, "relu", "he"), (10, 1, "linear", "he")]
        self.model.construct(layers_info, learning_rate=0.01)
        self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=1, 
            batch_size=32,
            wd=0.015,
            loss_function="huber:2",            
            graphical=False,
            log_plot=False
        )
        
        predictions = self.model.predict(self.X_test)
        
        # Check shape: should be (n_samples, n_outputs)
        self.assertEqual(predictions.shape, (self.X_test.shape[0], 1))

    def test_predict_returns_numpy_array(self):
        """Test that predict returns a numpy array"""
        layers_info = [(5, 10, "relu", "he"), (10, 1, "linear", "he")]
        self.model.construct(layers_info, learning_rate=0.01)
        self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=1, 
            batch_size=32,
            wd=0.02,
            loss_function="mae",
            graphical=False,
            log_plot=False
        )
        
        predictions = self.model.predict(self.X_test)
        
        self.assertIsInstance(predictions, np.ndarray)

    def test_multiple_layer_configuration(self):
        """Test model with different layer configurations"""
        layers_info = [
            (5, 16, "relu", "he"),
            (16, 8, "sigmoid", "random"),
            (8, 1, "linear", "xavier")
        ]
        self.model.construct(layers_info, learning_rate=0.01)
    
    def test_softmax(self):
        """Test classification model with softmax activation"""
        layers_info = [(5, 10, "relu", "he"), (10, 2, "softmax", "he")]
        y_train = []
        import random
        for i in range(len(self.X_train)):
            y_rand = random.choice(([0, 1], [1, 0]))
            y_train.append(y_rand)
        y_train = np.array(y_train)
        y_train = y_train.reshape(-2, 2)
        self.model.construct(layers_info, learning_rate=0.01)
        self.model.fit(
            self.X_train, 
            y_train, 
            epochs=1, 
            batch_size=32,
            wd=0.02,
            graphical=False,
            log_plot=False
        )

        predictions = self.model.predict(self.X_test)
        
        self.assertIsInstance(predictions, np.ndarray)

class TestCNN(unittest.TestCase):
    """Test suite for CNN class"""

    def setUp(self):
        """Initialize test fixtures before each test"""
        self.model = CNN(gpu=False)
        np.random.seed(42)
        self.X_train = np.random.randn(8, 15, 15, 3)
        self.y_train = np.random.randn(8, 4)
        self.X_test = np.random.randn(4, 15, 15, 3)
        self.y_test = np.random.randn(4, 4)

        y_clip = []
        for yi in self.y_train:
            one_hot = np.zeros_like(yi)
            one_hot[np.argmax(yi)] = 1
            y_clip.append(one_hot)
        self.y_train = np.asarray(y_clip)

    def test_model_initialization(self):
        """Test that model initializes correctly"""
        self.assertIsNotNone(self.model)

    def test_construct_sets_learning_rate(self):
        """Test that construct sets learning rate"""
        input_size = self.model.get_input_size(self.X_train)
        num_neurons = input_size//2 + self.y_train.shape[1]
        layers_info = [(input_size, num_neurons, "relu", "he"), (num_neurons, self.y_train.shape[1], "softmax", "he")]
        learning_rate = 0.001
        self.model.construct(layers_info, learning_rate=learning_rate)
        
        self.assertEqual(self.model.learning_rate, learning_rate)

    def test_fit_trains_without_error(self):
        """Test that fit method trains the model without errors"""
        input_size = self.model.get_input_size(self.X_train)
        num_neurons = input_size//2 + self.y_train.shape[1]
        layers_info = [(input_size, num_neurons, "relu", "he"), (num_neurons, self.y_train.shape[1], "softmax", "he")]
        self.model.construct(layers_info, learning_rate=0.01)
        
        # Should not raise any exception
        try:
            self.model.fit(
                self.X_train, 
                self.y_train, 
                epochs=2, 
                batch_size=32
            )
        except Exception as e:
            self.fail(f"fit() raised {type(e).__name__} unexpectedly: {e}")

    def test_predict_returns_correct_shape(self):
        """Test that predict returns correct output shape"""
        input_size = self.model.get_input_size(self.X_train)
        num_neurons = input_size//2 + self.y_train.shape[1]
        layers_info = [(input_size, num_neurons, "relu", "he"), (num_neurons, self.y_train.shape[1], "softmax", "he")]
        self.model.construct(layers_info, learning_rate=0.01)
        self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=1, 
            batch_size=32,
            report=True,
            graphical=True
        )
        
        predictions = self.model.predict(self.X_test)
        
        # Check shape
        self.assertEqual(predictions.shape, self.y_test.shape)

    def test_predict_returns_numpy_array(self):
        """Test that predict returns a numpy array"""
        input_size = self.model.get_input_size(self.X_train)
        num_neurons = input_size//2 + self.y_train.shape[1]
        layers_info = [(input_size, num_neurons, "relu", "he"), (num_neurons, self.y_train.shape[1], "softmax", "he")]
        self.model.construct(layers_info, learning_rate=0.01)
        self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=1, 
            batch_size=32,
        )
        
        predictions = self.model.predict(self.X_test)
        
        self.assertIsInstance(predictions, np.ndarray)

class TestRecurrentModel(unittest.TestCase):
    """Testing of recurrent model"""
    
    def __init__(self):
        self.X, self.y = self.generate_sine_wave_dataset(
            num_samples=3000,
            seq_len=30,
            freq=0.03,
            noise=0.0,
            many_to_one=True
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, split_test_percentage=0.3)

    def generate_sine_wave_dataset(
        num_samples=2000,
        seq_len=20,
        freq=0.05,
        noise=0.0,
        many_to_one=True
    ):
        """
        Generates a sine wave dataset for RNN testing.
        """

        # 1. Generate raw sine wave
        x = np.arange(num_samples + seq_len)
        y = np.sin(2 * np.pi * freq * x)

        # Optional noise
        if noise > 0:
            y += noise * np.random.randn(len(y))

        # 2. Build sequences
        X = []
        Y = []

        for i in range(num_samples):
            seq = y[i : i + seq_len]
            X.append(seq)

            if many_to_one:
                # Predict next value
                Y.append(y[i + seq_len])
            else:
                # Predict the whole sequence
                Y.append(seq)

        X = np.array(X).reshape(num_samples, seq_len, 1)   # (batch, seq_len, input_dim)
        Y = np.array(Y).reshape(num_samples, -1, 1) if not many_to_one else np.array(Y).reshape(num_samples, 1)

        return X.astype(np.float32), Y.astype(np.float32)
    
    def test_vanilla_w_o_RMSNorm(self):
        model = RecurrentModel()
        model.construct(
            input_dim=1,
            hidden_dim=32,
            output_dim=1,
            learning_rate= 1e-3,
            activation_function="linear",
            act_function_rnn="tanh",
            many_to_one=True,
            normalization=False
        )

        model.fit(self.X_train, self.y_train, epochs=15, batch_size=32, loss_function="mse", graphical=False)

    def test_vanilla_w_RMSNorm(self):
        model = RecurrentModel()
        model.construct(
            input_dim=1,
            hidden_dim=32,
            output_dim=1,
            learning_rate= 1e-3,
            activation_function="linear",
            act_function_rnn="tanh",
            many_to_one=True,
            normalization=True
        )

        model.fit(self.X_train, self.y_train, epochs=15, batch_size=32, loss_function="mse", graphical=False)
    
    def test_GRU_w_o_RMSNorm(self):
        model_gru = RecurrentModel(rnn_type="gru")
        model_gru.construct(
            input_dim=1,
            hidden_dim=32,
            output_dim=1,
            learning_rate= 1e-4,
            activation_function="linear",
            act_function_rnn="tanh",
            many_to_one=True,
            normalization = False
        )

        model_gru.fit(self.X_train, self.y_train, epochs=15, batch_size=32, loss_function="mse", graphical=False)

    def test_GRU_w_RMSNorm(self):
        model_gru = RecurrentModel(rnn_type="gru")
        model_gru.construct(
            input_dim=1,
            hidden_dim=32,
            output_dim=1,
            learning_rate= 1e-4,
            activation_function="linear",
            act_function_rnn="tanh",
            many_to_one=True,
            normalization = True
        )

        model_gru.fit(self.X_train, self.y_train, epochs=15, batch_size=32, loss_function="mse", graphical=False)

    def test_LSTM_w_o_RMSNorm(self):
        model_lstm = RecurrentModel(rnn_type="lstm")
        model_lstm.construct(
            input_dim=1,
            hidden_dim=32,
            output_dim=1,
            learning_rate= 1e-4,
            activation_function="linear",
            act_function_rnn="tanh",
            many_to_one=True,
            normalization=False
        )

        model_lstm.fit(self.X_train, self.y_train, epochs=15, batch_size=32, loss_function="mse", graphical=False)

    def test_LSTM_w_RMSNorm(self):
        model_lstm = RecurrentModel(rnn_type="lstm")
        model_lstm.construct(
            input_dim=1,
            hidden_dim=32,
            output_dim=1,
            learning_rate= 1e-4,
            activation_function="linear",
            act_function_rnn="tanh",
            many_to_one=True,
            normalization=True
        )

        model_lstm.fit(self.X_train, self.y_train, epochs=15, batch_size=32, loss_function="mse", graphical=False)

if __name__ == '__main__':
    unittest.main()