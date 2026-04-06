# SaANN - Self-automated Artificial Neural Network

[![Tests](https://github.com/ale99bra/SaANN/workflows/Run%20Tests/badge.svg)](https://github.com/ale99bra/SaANN/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A from-scratch implementation of an Artificial Neural Network (ANN) with multi-layer perceptron architecture. SaANN provides both manual and automatic workflows for building, training, and evaluating neural networks without relying on high-level frameworks like TensorFlow or PyTorch.

Made as a personal project for learning purposes.

## Features

- **Pure NumPy Implementation**: Built from scratch with NumPy for transparency and educational value
- **Flexible Architecture**: Support for custom layer configurations, activation functions, and initialization strategies
- **Dual Workflows**:
  - **Manual Mode**: Fine-grained control over every step
  - **Automatic Mode**: One-line training with preprocessing, scaling, and evaluation
- **Rich Visualization**: Real-time training plots, loss tracking, and prediction comparisons
- **Multiple Activation Functions**: ReLU, Sigmoid, Tanh, Linear, and Softmax
- **Advanced Training Options**: Batch training, configurable learning rates
- **Multiple Initialization Strategies**: He, Xavier, and random initialization

## Installation

### From GitHub

```bash
git clone https://github.com/ale99bra/SaANN.git
cd SaANN
pip install -e .
```

Or install directly:

```bash
pip install git+https://github.com/ale99bra/SaANN.git
```

### Requirements

- Python 3.8+
- NumPy >=1.0,<3.0
- Pandas >=1.0,<4.0
- Matplotlib >=3.7,<4.0

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Manual Workflow

For fine-grained control over training:

```python
from saann.models import SequentialModel
import numpy as np

# Assume X_train, y_train, X_test, y_test are prepared
model = SequentialModel()

# Define layer specifications: (input_size, neurons, activation, initialization)
layer_info = [
    (X_train.shape[1], 10, "relu", "he"),
    (10, 1, "linear", "he")
]

# Build and train
model.construct(layer_info, learning_rate=0.01)
model.fit(X_train, y_train, epochs=1000, batch_size=32, wd = 0.01, graphical=False, real_time=False, log_plot=False)

# Make predictions
y_pred = model.predict(X_test)
```

### Automatic Workflow

One-line training with automatic preprocessing and evaluation:

```python
from saann.models import SequentialModel
import numpy as np

# Assume X and y are your full dataset
model = SequentialModel()

layer_info = [
    (X.shape[1], 10, "relu", "he"),
    (10, 1, "linear", "he")
]

y_pred, final_pred_train, X_train, X_test, y_train, y_test = model.automatic(
    X=X,
    y=y,
    layers_info=layer_info,
    learning_rate=0.01,
    epochs=1000,
    batch_size=32,
    wd=0.01,
    split_test_percentage=0.3,
    scaling='minmax',
    graphical=True,
    real_time=False,
    log_plot=True,
    test_loss=True,
    scatter_comparison=True
)
```

## Project Structure

```
SaANN/
├── README.md                                       # This file
├── LICENSE                                         # MIT License
├── requirements.txt                                # Project dependencies
├── setup.py                                        # Installation configuration
├── .gitignore
│
└── saann/                                          # Main package
    ├── __init__.py
    ├── activation_functions.py                     # Activation functions (ReLU, Sigmoid, etc.)
    ├── gradients.py                                # Gradient computations
    ├── initiations.py                              # Weight initialization (He, Xavier, Random)
    ├── layers.py                                   # Layer implementations
    ├── losses.py                                   # Loss functions
    ├── models.py                                   # Sequential model class
    └── processing.py                               # Data preprocessing utilities
│
└── examples/                                       # Examples
    ├── automatic_diabetes_dataset_example.ipynb    # Example of automatic workflow
    ├── diabetes_dataset_example.ipynb              # Example of manual workflow
    └── XOR_example.ipynb                           # Example of manual workflow
│
└── .github/workflows                               
    └── tests.tml                                   # For running tests
│
└── tests
    ├── __init__.py  
    ├── test_activations.py                         # Test script for activation functions
    ├── test_layers.py                              # Test script for DenseLayer class
    └── test_models.py                              # Test script for model                       
```

## API Documentation

### SequentialModel

Main class for building and training neural networks.

#### Methods

**`construct(layers_info, learning_rate)`**

Builds the network architecture.

- `layers_info` (list): List of tuples `(input_size, neurons, activation, initialization)`
- `learning_rate` (float): Learning rate for optimization

**`fit(X, y, epochs, batch_size, wd, graphical, real_time, log_plot)`**

Trains the model on data.

- `X` (array): Training features
- `y` (array): Training labels
- `epochs` (int): Number of training epochs
- `batch_size` (int): Batch size for training
- `wd` (float): Weight decay factor for regularization
- `graphical` (bool): Display training plot
- `real_time` (bool): Update training plot in real-time
- `log_plot` (bool): Display plot in a semilogy scale

**`predict(X)`**

Makes predictions on new data.

- `X` (array): Input features
- Returns: Predictions (array)

**`automatic(X, y, layers_info, learning_rate, epochs, batch_size, wd, split_test_percentage, scaling, graphical, real_time, log_plot, test_loss, scatter_comparison)`**

Runs the complete pipeline: data splitting, scaling, training, and evaluation.

- `X` (array): Full dataset features
- `y` (array): Full dataset labels
- `layers_info` (list): List of tuples `(input_size, neurons, activation, initialization)`
- `learning_rate` (float): Learning rate for optimization
- `epochs` (int): Number of training epochs
- `batch_size` (int): Batch size for training
- `wd` (float): Weight decay factor for regularization
- `split_test_percentage` (float): Test/Train split ratio (0.0-1.0)
- `scaling` (str): Scaling method ('zscore', 'minmax', 'log', 'mean' or None)
- `graphical` (bool): Display training plot
- `real_time` (bool): Update training plot in real-time
- `log_plot` (bool): Display plot in a semilogy scale
- `test_loss` (bool): Prints the loss of the predicted and test data
- `scatter_comparison` (bool): Display the scatter plot Test vs. Predicted
- Returns: Tuple of (predictions, train_predictions, X_train, X_test, y_train, y_test)

### Scaling

Class for applying scaling to your dataset

#### Methods

**`zScore(x)`**

Applies a Z-score standardization

- `x` (array): Features to scale

**`MinMax(x)`**

Applies a Min-Max standardization

- `x` (array): Features to scale

**`LogNorm(x)`**

Applies a natural log transformation

- `x` (array): Features to scale

**`MeanNorm(x)`**

Applies a a mean normalization

- `x` (array): Features to scale

### Automatic Train/Test split function

**`train_test_split(X, y, split_test_percentage)`**

Splits the dataset into Train and Test sets

- `X` (array): Full dataset features
- `y` (array): Full dataset labels
- `split_test_percentage` (float): Test/Train split ratio (0.0-1.0)

### Loss functions

Currently, the model utilizes the Mean Squared Error (MSE) for the training (in the future, other functions will be added).
However, it is possible to call both the MSE and the Mean Absolute Error (MAE) functions for personal uses as well

**`MSE(y_true, y_pred)`**

Calculates the Mean Squared Error between the predicted and testing data

- `y_true` (array): Testing labels
- `y_pred` (array): Predicted labels

**`MAE(y_true, y_pred)`**

Calculates the Mean Squared Error between the predicted and testing data

- `y_true` (array): Testing labels
- `y_pred` (array): Predicted labels


## Supported Activation Functions

- **relu**: Rectified Linear Unit
- **sigmoid**: Sigmoid function
- **tanh**: Hyperbolic tangent
- **linear**: Linear activation (identity)
- **softmax**: Softmax activation

## Initialization Strategies

- **he**: He initialization (recommended for ReLU)
- **xavier**: Xavier/Glorot initialization
- **random**: Random normal initialization

## Examples

See the `examples/` directory for complete working examples:

- `diabetes_dataset_example.ipynb`: Manual workflow using `scikit-learn` diabetes dataset with visualization
- `automatic_diabetes_dataset_example.ipynb`: Automatic workflow using `scikit-learn` diabetes dataset with visualization
- `XOR_example.ipynb`: Testing of the capabilities of the non-linearity application

## Architecture Overview

SaANN implements a Multi-Layer Perceptron (MLP) with:

- Forward propagation with configurable activations
- Backpropagation for gradient computation
- Batch gradient descent optimization
- Flexible layer configuration

## Performance Considerations

- **Best for**: Educational purposes, small to medium datasets
- **Limitations**: Not optimized for GPU computation; slower than production frameworks on large datasets
- **Use TensorFlow/PyTorch for**: Production systems, very large datasets, complex architectures

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests (be aware that this is mostly a learning project).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENCE) file for details.

## Author

**Alessio Branda** - [GitHub Profile](https://github.com/ale99bra)

## Acknowledgments

- Educational implementation inspired by neural network fundamentals
- Built to understand deep learning from first principles

---

## Getting Help

If you encounter any issues or have questions:

1. Check the `examples/` directory for usage patterns
2. Review the inline code documentation
3. Open an issue on GitHub with a detailed description