# SaANN — Self-automated Artificial Neural Network

[![Tests](https://github.com/ale99bra/SaANN/workflows/Run%20Tests/badge.svg)](https://github.com/ale99bra/SaANN/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational deep-learning framework built from scratch with NumPy/CuPy. SaANN provides transparent implementations of MLPs and CNNs, with optional GPU acceleration and comprehensive metrics for learning and experimentation.

**⚠️ Important**: SaANN is designed for learning, not production.

## Features

### 🌐 Core Architecture

- **Pure NumPy Implementation**: Built entirely from scratch for transparency and educational value
- **Optional CuPy GPU Backend**: Transparent GPU acceleration without code changes
- **Flexible Architecture**: Support for custom layer configurations, activation functions, and initialization strategies
- **Dual Workflows**:
  - **Manual Mode**: Fine-grained control over every step
  - **Automatic Mode**: One-line training with preprocessing, scaling, and evaluation

### 🧩 Multi-Layer Perceptron (MLP)

- Multiple activation functions: ReLU, Sigmoid, Tanh, Linear, Softmax
- Multiple loss functions: MSE, MAE, Huber, CE (with regularization)
- Batch training with configurable learning rates
- Advanced options: Batch Normalization, Dropout, Weight Decay
- Initialization strategies: He, Xavier, Random

### 🧬 Convolutional Neural Networks (CNN) — Experimental but Fully Functional

- Custom convolution layers with configurable filters
- im2col/col2im implementation for efficient convolution
- Max-pooling layers
- BatchNorm2D for training stability
- MLP classifier head
- Full GPU acceleration support
- Integrated metrics reporting

### 💾 Model Management

- `save_model(path)`: Save models in portable format
- `load_model(path)`: Auto-detect and load any SaANN model
- Architecture reconstruction with versioning
- Prevents incompatible model loading

### 📊 Comprehensive Metrics Suite

- **Classification Metrics**: Precision, Recall, F1 (Macro/Micro/Weighted)
- **Advanced Metrics**: Balanced Accuracy, Specificity, Matthews Correlation Coefficient (MCC)
- **Visualization**: Confusion matrices, ROC curves (One-vs-Rest)
- **Reporting**: `Metrics.report()` for full summaries

### 📈 Training & Evaluation

- Real-time training plots and loss tracking
- Automatic train/test split
- Scaling utilities: Z-score, Min-Max, Log, Mean normalization
- Prediction comparisons and scatter plots

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

- Python 3.11+
- NumPy >=2.0,<3.0
- Pandas >=3.0,<4.0
- Matplotlib >=3.10,<4.0
- Pillow >=12.0,<13.0

Install dependencies:

```bash
pip install -r requirements.txt
```

### Optional: GPU Support (CuPy)

For GPU acceleration with CUDA 12.x:

```bash
pip install saann[gpu]
```

Or manually:

```bash
pip install cupy-cuda12x>=14.0
```

SaANN automatically switches between NumPy and CuPy based on configuration.

## Quick Start

### MLP Example (SequentialModel)

For fine-grained control over training:

```python
from saann.models import SequentialModel
import numpy as np

# Assume X_train, y_train, X_test, y_test are prepared
model = SequentialModel(gpu=False)

# Define layer specifications: (input_size, neurons, activation, initialization)
layer_info = [
    (X_train.shape[1], 10, "relu", "he"),
    (10, 1, "linear", "he")
]

# Build and train
model.construct(layer_info, learning_rate=0.01)
model.fit(X_train, y_train, epochs=1000, batch_size=32, wd=0.01, loss_function='MSE', graphical=False)

# Make predictions
y_pred = model.predict(X_test)
```

### CNN Example — Flower Classification

Classify flowers using the Kaggle flowers dataset (daisy, dandelion, rose, sunflower, tulip):

#### Dataset Preparation

```python
from saann.processing import ImageProcessing

IP = ImageProcessing(images_path="path/to/flowers")
X_train, X_test, y_train, y_test, list_classes = IP.ready_dataset(
    size=98,
    amount=100,
    shuffle=True,
    remove_resized=True,
    split_test_percentage=0.3
)
```

#### Training the CNN

```python
from saann.models import CNN

model_cnn = CNN(filter_size=3, num_filters=32, padding=1, stride=1, gpu=True)

input_size = model_cnn.get_input_size(X_train)

layer_info = [
    (input_size, 256, 'relu', 'he'),
    (256, 128, 'relu', 'he'),
    (128, y_train.shape[1], 'softmax', 'he')
]

model_cnn.construct(layers_info=layer_info, learning_rate=1e-4)

final_pred = model_cnn.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    wd=0.00001,
    graphical=True,
    report=True
)
```

#### Metrics Example Output

```
Macro F1: ~0.37
Weighted F1: ~0.36
AUC: ~0.62
MCC: ~0.04
```

These results are expected for a small dataset, 20 epochs, and an educational CNN.

### Model Saving & Loading

#### Save a trained model

```python
model_cnn.save_model("flower_cnn.pkl")
```

#### Load and evaluate

```python
from saann.models import load_model
from saann.metrics import Metrics

model = load_model("flower_cnn.pkl")

# ... prepare data ...
y_pred = model.predict(X_test)

metrics = Metrics(y_pred=y_pred, y_test=y_test)
metrics.report()
```

**⚠️ Note**: If the model was trained with GPU acceleration, GPU must be enabled when loading.

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
    ├── layers.py                                   # Layer implementations (Dense, BatchNorm, etc.)
    ├── losses.py                                   # Loss functions (MSE, MAE, Huber)
    ├── models.py                                   # Model classes (SequentialModel, CNN, load_model)
    ├── processing.py                               # Data preprocessing & ImageProcessing utilities
    ├── metrics.py                                  # Metrics suite (Precision, Recall, F1, ROC, etc.)
    └── backend.py                                  # NumPy/CuPy backend abstraction
│
└── examples/                                       # Examples
    ├── automatic_diabetes_dataset_example.ipynb    # Example of automatic workflow
    ├── diabetes_dataset_example.ipynb              # Example of manual workflow
    ├── XOR_example.ipynb                           # Example of manual workflow
    └── flower_cnn_example.ipynb                    # CNN training and evaluation
│
└── .github/workflows                               
    └── tests.yml                                   # GitHub Actions test workflow
│
└── tests/
    ├── __init__.py  
    ├── test_activations.py                         # Test script for activation functions
    ├── test_layers.py                              # Test script for layer classes
    ├── test_models.py                              # Test script for model classes
    └── test_metrics.py                             # Test script for metrics
```

## API Documentation

### *Models*

### SequentialModel (MLP)

Main class for building and training multi-layer perceptrons.

#### Methods

**`__init__(gpu)`**

Initialize a SequentialModel.

- `gpu` (bool): Enable GPU acceleration (requires CuPy)

**`construct(layers_info, learning_rate, batch_norm, dropout)`**

Builds the network architecture.

- `layers_info` (list): List of tuples `(input_size, neurons, activation, initialization)`
- `learning_rate` (float): Learning rate for optimization
- `batch_norm` (bool): Enables batch normalization
- `dropout` (bool): Enables dropout

**`fit(X, y, epochs, batch_size, wd, loss_function, graphical, real_time, log_plot)`**

Trains the model on data.

- `X` (array): Training features
- `y` (array): Training labels
- `epochs` (int): Number of training epochs
- `batch_size` (int): Batch size for training
- `wd` (float): Weight decay factor for regularization
- `loss_function` (str): 'MSE', 'MAE', or 'Huber' (or 'Huber:delta' for custom delta)
- `graphical` (bool): Display training plot
- `real_time` (bool): Update training plot in real-time
- `log_plot` (bool): Display plot in semilogy scale
- Returns: Final prediction of training (array)

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
- `loss_function` (str): 'MSE', 'MAE', or 'Huber'
- `split_test_percentage` (float): Test/Train split ratio (0.0-1.0)
- `scaling` (str): Scaling method ('zscore', 'minmax', 'log', 'mean', or None)
- `graphical` (bool): Display training plot
- `real_time` (bool): Update training plot in real-time
- `log_plot` (bool): Display plot in semilogy scale
- `test_loss` (bool): Print loss metrics
- `scatter_comparison` (bool): Display scatter plot of test vs. predicted
- Returns: Tuple of (predictions, train_predictions, X_train, X_test, y_train, y_test)

### CNN (Convolutional Neural Network)

Experimental but fully functional convolutional neural network for image classification.

#### Methods

**`__init__(filter_size, num_filters, padding, stride, gpu)`**

Initialize a CNN.

- `filter_size` (int): Size of convolution filters
- `num_filters` (int): Number of filters in the first conv layer
- `padding` (int): Padding for convolutions
- `stride` (int): Stride for convolutions
- `gpu` (bool): Enable GPU acceleration (requires CuPy)

**`construct(layers_info, learning_rate, batch_norm, dropout)`**

Builds the network architecture.

- `layers_info` (list): List of tuples `(input_size, neurons, activation, initialization)`
- `learning_rate` (float): Learning rate for optimization
- `batch_norm` (bool): Enables batch normalization
- `dropout` (bool): Enables dropout

Build the network architecture (MLP head after convolutions).

**`fit(X, y, epochs, batch_size, wd, graphical, report)`**

Train the CNN on data.

- `X` (array): Training images (batch_size, height, width, channels)
- `y` (array): Training labels
- `epochs` (int): Number of training epochs
- `batch_size` (int): Batch size for training
- `wd` (float): Weight decay factor
- `graphical` (bool): Display training plot
- `report` (bool): Print metrics report after training
- Returns: Final prediction of training (array)

**`predict(X)`**

Make predictions on new images.

- `X` (array): Input images
- Returns: Predictions (array)

**`save_model(path)`**

Save the model's architecture and weights in a pickle file
- `path` (str): File path for output

### `load_model(path)`

Load the model saved using the model.save_model(path) method. It automatically detects the type of model saved. 

- `path` (str): File path for output
- Returns: Model (class)

### *Metrics*

Comprehensive metrics suite for model evaluation.

### Metrics

#### Methods

**`__init__(y_pred, y_test)`**

Initialize metrics calculator.

- `y_pred` (array): Predicted labels
- `y_test` (array): True labels

**`report(graphical, threshold_step)`**

Print a full metrics summary including:

- Precision, Recall, F1 (Macro/Micro/Weighted)
- Balanced Accuracy, Specificity, MCC
- Confusion Matrix
- ROC curves (One-vs-Rest for multi-class)


- `graphical` (bool): Plots ROC curves and heatmap of the confusion matrix
- `threshold_step` (float): Step used to increment from 0 to 1 for ROC calculation

**`confusion_matrix(graphical)`**

Calculate the confusion matrix

- `graphical` (bool): Plots the heatmap of the confusion matrix
- Returns: Confusion matrix (array)

**`precision()`**

Calculate precision for each class

- Returns: Precisions (array)

**`recall()`**

Calculate sensitivity for each class

- Returns: Recalls (array)

**`F1score()`**

Calculate the F1-score for each class

- Returns: F1-scores (array)

**`macro_f1()`**

Calculate the Macro F1-score for each class

- Returns: Macro F1-scores (float)

**`micro_f1()`**

Calculate the Micro F1-score for each class

- Returns: Micro F1-scores (float)

**`weighted_f1()`**

Calculate the Weighted F1-score for each class

- Returns: Weighted F1-scores (float)

**`AUC(graphical, threshold_step)`**

Calculate the macro-averaged AUC

- `graphical` (bool): Plots ROC curves
- `threshold_step` (float): Step used to increment from 0 to 1 for ROC calculation
- Returns: Macro-averaged AUC (float)

**`balanced_accuracy()`**

Calculate balanced accuracy for each class

- Returns: Balanced accuracy (array)

**`specificity()`**

Calculate specificity for each class

- Returns: Specificity (array)

**`cohen_kappa()`**

Calculate Cohen's kappa for each class

- Returns: Cohen's kappa (float)

**`MCC()`**

Calculate Matthews correlation coefficient for each class

- Returns: Matthews correlation coefficient (float)

### *Processing*

**`train_test_split(X, y, split_test_percentage)`**

Splits the dataset provided into Train and Test arrays

- `X` (array): Features
- `y` (array): Targets
- `split_test_percentage` (float): Test/Train split ratio
- Returns: Tuple of (X_train, X_test, y_train, y_test)


### ImageProcessing

Utilities for image dataset preparation.

#### Methods

**`ready_dataset(size, amount, shuffle, remove_resized, split_test_percentage)`**

Load and prepare an image dataset.

- `size` (int): Resize images to size×size
- `amount` (int): Maximum images per class. Can be None.
- `shuffle` (bool): Shuffle the dataset
- `remove_resized` (bool): Remove resized files after processing
- `split_test_percentage` (float): Test/Train split ratio
- Returns: Tuple of (X_train, X_test, y_train, y_test, class_names)

### Scaling

Data scaling utilities.

#### Methods

**`zScore(x)`, `MinMax(x)`, `LogNorm(x)`, `MeanNorm(x)`**

Apply respective scaling transformations to features.

- `x` (array): Features to scale
- Returns: Scaled features (array)

### *Losses*

**`MSE(y_true, y_pred)`**

Mean Squared Error loss.

- `y_true` (array): Targets used for testing
- `y_pred` (array). Predicted targets

**`MAE(y_true, y_pred)`**

Mean Absolute Error loss.

- `y_true` (array): Targets used for testing
- `y_pred` (array). Predicted targets

**`cross_entropy(y_true, y_pred)`**

Cross-Entropy Error loss. 

- `y_true` (array): Targets used for testing
- `y_pred` (array). Predicted targets

**`Huber(y_true, y_pred, delta)`**

Huber loss with configurable delta parameter.

- `y_true` (array): Targets used for testing
- `y_pred` (array). Predicted targets

**`R2_score(y_true, y_pred)`**

Coefficient of determination (R²).

- `y_true` (array): Targets used for testing
- `y_pred` (array). Predicted targets

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

- `diabetes_dataset_example.ipynb`: Manual MLP workflow
- `automatic_diabetes_dataset_example.ipynb`: Automatic MLP workflow
- `XOR_example.ipynb`: Non-linearity testing
- `flower_cnn_example.ipynb`: CNN training and evaluation with metrics

## Architecture Overview

SaANN implements:

- **Multi-Layer Perceptron (MLP)**: Forward/backpropagation, batch gradient descent
- **Convolutional Neural Network (CNN)**: Conv2D → MaxPool → BatchNorm2D → MLP head
- **Training**: Configurable optimizers, learning rates, regularization
- **Backend**: Transparent NumPy/CuPy switching

## Performance Considerations

- **Best for**: Educational purposes, understanding deep learning internals, small to medium datasets
- **GPU Acceleration**: Significantly faster with `gpu=True` and CuPy installed
- **im2col/col2im**: Slower than optimized libraries like cuDNN
- **Limitations**: Not optimized for production; designed for learning
- **Use TensorFlow/PyTorch for**: Production systems, very large datasets, complex architectures

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests. Remember that this is primarily an educational personal project.

## License

This project is licensed under the MIT License - see the [LICENSE](/LICENCE) file for details.

## Author

**Alessio Branda** - [GitHub Profile](https://github.com/ale99bra)

## Acknowledgments

- Educational implementation inspired by neural network fundamentals
- Built to understand deep learning from first principles
- GPU acceleration via CuPy for transparent performance optimization

---

## Getting Help

If you encounter any issues or have questions:

1. Check the `examples/` directory for usage patterns
2. Review the inline code documentation
3. Open an issue on GitHub with a detailed description