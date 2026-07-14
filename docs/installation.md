# Installation

## From GitHub

```bash
git clone https://github.com/ale99bra/SaANN.git
cd SaANN
pip install -e .
```

Or install directly:

```bash
pip install git+https://github.com/ale99bra/SaANN.git
```

## Requirements

- Python 3.11+
- NumPy >=2.0,<3.0
- Pandas >=3.0,<4.0
- Matplotlib >=3.10,<4.0
- Pillow >=12.0,<13.0

Install dependencies:

```bash
pip install -r requirements.txt
```

## Optional: GPU Support (CuPy)

SaANN supports optional GPU acceleration through CuPy, which provides a NumPy‑compatible API backed by CUDA.
GPU support is not installed by default.

To install SaANN with GPU extras:

```bash
pip install saann[gpu]
```

The `gpu` extra installs CuPy bindings, but you must still install the correct CUDA‑enabled CuPy wheel (e.g., `cupy-cuda12x[ctk]`) depending on your system.
For more info, refer to [CuPy Installation](https://docs.cupy.dev/en/stable/install.html).

SaANN automatically switches between NumPy and CuPy based on configuration.

---

⬅ [Back to README](../README.md) · Next: [Quick Start](quickstart.md)
