# Architecture

## Project Structure

```
SaANN/
├── README.md                                       # README markdown
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
    ├── backend.py                                  # NumPy/CuPy backend abstraction
    ├── tokenizer.py                                # Tokenizers for Transformer
    ├── training.py                                 # Training functions for Transformer
    └── finance/
        ├──__init__.py
        ├── datasets.py                             # Dataset preparation for RNN training
        ├── features.py                             # Volatility features
        ├── metrics.py                              # Metrics for evaluation
        ├── sequence.py                             # Sequence builder
        ├── targets.py                              # Forward realised volatility target
        └── volatility.py                           # Volatility estimator
    └── transformer/
        ├──__init__.py
        ├── attention.py                            # Attention algorithm
        ├── blocks.py                               # Blocks for transformer head
        ├── embeddings.py                           # Positional Embedding layer for Transformers
        └── transformer_model.py                    # GPT-style Transformer model
│
└── examples/                                       # Examples
    ├── automatic_diabetes_dataset_example.ipynb    # Example of MLP automatic workflow
    ├── diabetes_dataset_example.ipynb              # Example of MLP manual workflow
    ├── XOR_example.ipynb                           # Example of MLP manual workflow
    ├── flower_cnn_example.ipynb                    # CNN training and evaluation - flowers
    ├── EO_cnn_example.ipynb                        # CNN training and evaluation - satellite images
    └── transformer_dummy_example.ipynb             # Example of Transformer API
│
└── .github/workflows
    └── tests.yml                                   # GitHub Actions test workflow
│
└── tests/
    ├── __init__.py
    ├── test_activations.py                         # Test script for activation functions
    ├── test_finance.py                             # Test script for finance module
    ├── test_layers.py                              # Test script for layer classes
    ├── test_metrics.py                             # Test script for metrics
    ├── test_models.py                              # Test script for model classes
    └── test_transformer.py                         # Test script for transformer classes
│
└── docs/
    ├── api-reference.md                            # API documentation
    ├── architecture.md                             # This file
    ├── features.md                                 # Rundown of the models
    ├── finance.md                                  # Rundown of the finance module
    ├── installation.md                             # Installation guide
    └── quickstart.md                               # Get started guide
```

## Architecture Overview

SaANN implements:

- **Multi-Layer Perceptron (MLP)**: Forward/backpropagation, batch gradient descent, experimental Cross-Training
- **Convolutional Neural Network (CNN)**: Conv2D → MaxPool → BatchNorm2D → MLP head
- **Recurrent Neural Network (RNN)**: Vanilla, GRU, LSTM
- **Transformer**
- **Training**: Configurable optimizers, learning rates, regularization
- **Backend**: Transparent NumPy/CuPy switching
- **Financial Forecasting Module**: Volatility estimation, HAR feature engineering, Sequence generation, Forecasting metrics, RNN ready datasets

## Performance Considerations

- **Best for**: Educational purposes, understanding deep learning internals, small to medium datasets
- **GPU Acceleration**: Significantly faster with `gpu=True` and CuPy installed
- **im2col/col2im**: Slower than optimized libraries like cuDNN
- **Limitations**: Not optimized for production; designed for learning
- **Use TensorFlow/PyTorch for**: Production systems, very large datasets, complex architectures

---

⬅ Previous: [API Reference](api-reference.md) · [Back to README](../README.md)
