import unittest
import numpy as np
from saann.layers import DenseLayer


class TestLayer(unittest.TestCase):
    """Test suite for Layer class"""

    def setUp(self):
        """Initialize test fixtures"""
        np.random.seed(42)
        self.input_size = 5
        self.output_size = 10
        self.batch_size = 32

    def test_layer_initialization(self):
        """Test that layer initializes correctly"""
        layer = DenseLayer(
            extras=[True, True],
            num_inputs=self.input_size,
            num_neurons=self.output_size,
            activation_function="relu",
            init_function="he"
        )
        
        self.assertIsNotNone(layer)
        self.assertEqual(layer.weights.shape, (self.input_size, self.output_size))
        self.assertEqual(layer.biases.shape, (1, self.output_size))

    def test_layer_forward_pass(self):
        """Test layer forward pass"""
        layer = DenseLayer(
            extras=[False, True],
            num_inputs=self.input_size,
            num_neurons=self.output_size,
            activation_function="relu",
            init_function="he"
        )
        
        X = np.random.randn(self.batch_size, self.input_size)
        output = layer.forward(X)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_size))

    def test_layer_backward_pass(self):
        """Test layer backward pass"""
        layer = DenseLayer(
            extras=[True, False],
            num_inputs=self.input_size,
            num_neurons=self.output_size,
            activation_function="relu",
            init_function="he"
        )
        
        X = np.random.randn(self.batch_size, self.input_size)
        output = layer.forward(X)
        
        # Create mock gradient from previous layer
        dL_dA = np.random.randn(self.batch_size, self.output_size)
        
        # Backward pass should not raise error
        try:
            dL_dX = layer.backward(dL_dA)
            self.assertEqual(dL_dX.shape, X.shape)
        except Exception as e:
            self.fail(f"backward() raised {type(e).__name__} unexpectedly: {e}")

    def test_layer_weight_shapes(self):
        """Test that layer weights have correct shapes"""
        layer = DenseLayer(
            extras=[False, False],
            num_inputs=self.input_size,
            num_neurons=self.output_size,
            activation_function="sigmoid",
            init_function="xavier"
        )
        
        self.assertEqual(layer.weights.shape, (self.input_size, self.output_size))
        self.assertEqual(layer.biases.shape, (1, self.output_size))

    def test_different_activations(self):
        """Test layer with different activation functions"""
        activations = ["relu", "sigmoid", "tanh", "linear"]
        
        for activation in activations:
            with self.subTest(activation=activation):
                layer = DenseLayer(
                    extras=[False, False],
                    num_inputs=self.input_size,
                    num_neurons=self.output_size,
                    activation_function=activation,
                    init_function="random",
                    random_scale=0.01
                )
                
                X = np.random.randn(self.batch_size, self.input_size)
                output = layer.forward(X)
                
                self.assertEqual(output.shape, (self.batch_size, self.output_size))


if __name__ == '__main__':
    unittest.main()