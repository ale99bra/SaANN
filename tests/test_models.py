import unittest
import numpy as np
from saann.models import SequentialModel, CNN


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

if __name__ == '__main__':
    unittest.main()