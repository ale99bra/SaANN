import unittest
import numpy as np
from saann.models import SequentialModel


class TestSequentialModel(unittest.TestCase):
    """Test suite for SequentialModel class"""

    def setUp(self):
        """Initialize test fixtures before each test"""
        self.model = SequentialModel()
        np.random.seed(42)
        self.X_train = np.random.randn(100, 5)
        self.y_train = np.random.randn(100, 1)
        self.X_test = np.random.randn(20, 5)

    def test_model_initialization(self):
        """Test that model initializes correctly"""
        self.assertIsNotNone(self.model)

    def test_construct_creates_layers(self):
        """Test that construct method creates layers"""
        layers_info = [(5, 10, "relu", "he"), (10, 1, "linear", "he")]
        self.model.construct(layers_info, learning_rate=0.01)
        
        # Verify layers were created
        self.assertIsNotNone(self.model.layers)
        self.assertGreater(len(self.model.layers), 0)

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
            graphical=False,
            log_plot=False
        )
        
        predictions = self.model.predict(self.X_test)
        
        self.assertIsInstance(predictions, np.ndarray)

    def test_multiple_layer_configuration(self):
        """Test model with different layer configurations"""
        layers_info = [
            (5, 16, "relu", "he"),
            (16, 8, "relu", "he"),
            (8, 1, "linear", "he")
        ]
        self.model.construct(layers_info, learning_rate=0.01)
        
        self.assertGreater(len(self.model.layers), 0)


if __name__ == '__main__':
    unittest.main()