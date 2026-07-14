# SaANN — Self-automated Artificial Neural Network

[![Tests](https://github.com/ale99bra/SaANN/workflows/Run%20Tests/badge.svg)](https://github.com/ale99bra/SaANN/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational deep-learning framework built from scratch with NumPy/CuPy. SaANN provides transparent implementations of MLPs, CNNs and RNNs, with optional GPU acceleration and comprehensive metrics for learning and experimentation.

*Major update*: VERSION 0.3.0
- Added experimental feature: Cross‑Training MLPs — training of two parallel MLPs as a way to regularize early training
- Added TransformerModel — see [docs/features.md](docs/features.md#-transformermodel-gptstyle-decoderonly-transformer)
    - Tests not yet implemented

**⚠️ Important**: SaANN is designed for personal learning, not production. This project was created as an exercise to deepen my understanding of neural networks. While others are welcome to explore or use it, its primary purpose is educational for myself.

## Documentation

Full documentation lives in [`docs/`](docs/):

- [Installation](docs/installation.md) — Setup, requirements, optional GPU (CuPy) support
- [Quick Start](docs/quickstart.md) — Copy‑paste examples for every model type
- [API Reference](docs/api-reference.md) — Full method-by-method documentation
- [Architecture](docs/architecture.md) — Project structure and design overview
- [Features](docs/features.md) — Core architecture, MLP, CNN, RNN, Transformer, Cross-Training, metrics


## Installation

```bash
git clone https://github.com/ale99bra/SaANN.git
cd SaANN
pip install -e .
```

See [docs/installation.md](docs/installation.md) for requirements and GPU setup.

## Quick Start

```python
from saann.models import SequentialModel

model = SequentialModel(gpu=False)
layer_info = [
    (X_train.shape[1], 10, "relu", "he"),
    (10, 1, "linear", "he")
]
model.construct(layer_info, learning_rate=0.01)
model.fit(X_train, y_train, epochs=1000, batch_size=32, wd=0.01, loss_function='MSE')

y_pred = model.predict(X_test)
```

More examples (RNN, CNN, Transformer, Cross-Training) in [docs/quickstart.md](docs/quickstart.md).

## Examples

See the `examples/` directory for complete working notebooks:

- `diabetes_dataset_example.ipynb`: Manual MLP workflow
- `automatic_diabetes_dataset_example.ipynb`: Automatic MLP workflow
- `XOR_example.ipynb`: Non-linearity testing
- `flower_cnn_example.ipynb`: CNN training and evaluation with metrics

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests. Remember that this is primarily an educational personal project.

## License

This project is licensed under the MIT License - see the [LICENSE](/LICENCE) file for details.

## Author

**Alessio Branda** - [GitHub Profile](https://github.com/ale99bra)

## Getting Help

If you encounter any issues or have questions:

1. Check the [examples](examples/) directory for usage patterns
2. Review the [documentation](docs/)
3. Open an issue on GitHub with a detailed description
