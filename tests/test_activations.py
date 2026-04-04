import unittest
import numpy as np
from saann import activation_functions as af


class TestActivationFunctions(unittest.TestCase):
    """Test suite for activation functions"""

    def setUp(self):
        """Initialize test data"""
        self.X = np.array([-2, -1, 0, 1, 2], dtype=float)
        np.random.seed(42)

    def test_relu_activation(self):
        """Test ReLU activation function"""
        output = af.relu(self.X)
        expected = np.array([0, 0, 0, 1, 2], dtype=float)
        np.testing.assert_array_equal(output, expected)

    def test_relu_derivative(self):
        """Test ReLU derivative"""
        output = af.relu_derivative(self.X)
        expected = np.array([0, 0, 0, 1, 1], dtype=float)
        np.testing.assert_array_equal(output, expected)

    def test_sigmoid_activation(self):
        """Test sigmoid activation function"""
        output = af.sigmoid(np.array([0]))
        # sigmoid(0) should be 0.5
        self.assertAlmostEqual(output[0], 0.5, places=5)

    def test_sigmoid_range(self):
        """Test that sigmoid output is between 0 and 1"""
        X = np.linspace(-10, 10, 100)
        output = af.sigmoid(X)
        
        self.assertTrue(np.all(output >= 0))
        self.assertTrue(np.all(output <= 1))

    def test_tanh_range(self):
        """Test that tanh output is between -1 and 1"""
        X = np.linspace(-10, 10, 100)
        output = af.tanh(X)
        
        self.assertTrue(np.all(output >= -1))
        self.assertTrue(np.all(output <= 1))

    def test_linear_activation(self):
        """Test linear activation (identity function)"""
        output = af.linear(self.X)
        np.testing.assert_array_equal(output, self.X)

    def test_linear_derivative(self):
        """Test linear derivative (should be all ones)"""
        output = af.linear_derivative(self.X)
        expected = np.ones_like(self.X)
        np.testing.assert_array_equal(output, expected)


if __name__ == '__main__':
    unittest.main()