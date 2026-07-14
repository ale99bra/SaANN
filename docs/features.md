# Features

## 🌐 Core Architecture

- **Pure NumPy Implementation**: Built entirely from scratch for transparency and educational value
- **Optional CuPy GPU Backend**: Transparent GPU acceleration without code changes
- **Flexible Architecture**: Support for custom layer configurations, activation functions, and initialization strategies
- **Dual Workflows**:
  - **Manual Mode**: Fine-grained control over every step
  - **Automatic Mode**: One-line training with preprocessing, scaling, and evaluation

## 🧩 Multi-Layer Perceptron (MLP)

- Multiple activation functions: ReLU, Sigmoid, Tanh, Linear, Softmax
- Multiple loss functions: MSE, MAE, Huber, CE (with regularization)
- Batch training with configurable learning rates
- Advanced options: Batch Normalization, Dropout, Weight Decay
- Initialization strategies: He, Xavier, Random

## 🧬 Convolutional Neural Networks (CNN) — Experimental but Fully Functional

- Custom convolution layers with configurable filters
- im2col/col2im implementation for efficient convolution
- Max-pooling layers
- BatchNorm2D for training stability
- MLP classifier head
- Full GPU acceleration support
- Integrated metrics reporting

## 🧠 Recurrent Neural Networks (RNN, GRU, LSTM) — Sequence Modeling

- Pure NumPy/CuPy implementation
- Vanilla RNN, GRU, and LSTM cells
    - GRU and LSTM perform best on complex or long‑range sequence tasks
    - On simple periodic signals, a vanilla RNN may outperform them
- Many‑to‑one and many‑to‑many modes
    - `many_to_one=True`: only the final hidden state is passed to the Dense head
    - `many_to_one=False`: outputs at every timestep (sequence‑to‑sequence)
- Customizable hidden dimension, activation, and initialization
    - For GRU/LSTM, Xavier initialization is recommended for stability
- Full backpropagation through time (BPTT)
- Gradient clipping for stability
- Root-Mean-Squared Normalization (RMSNorm) included
    - RMSNorm helps stabilize training on long sequences or tasks with large activation variance
    - When using RMSNorm with GRU/LSTM, a smaller learning rate (e.g., 1e‑4) is recommended
        - SaANN already makes up for this by applying only 10% of the chosen learning rate to the scale-vector updates
- Compatible with GPU acceleration
- Dense head for regression or classification
- **⚠️ Important**: RNNs are still under development. Future updates might improve current performance

## 🧠 TransformerModel (GPT‑Style Decoder‑Only Transformer)

SaANN includes a fully‑from‑scratch implementation of a GPT‑style Transformer, designed for education, experimentation, and transparent deep‑learning research.
The implementation is backend‑agnostic and runs on CPU (NumPy) or GPU (CuPy) using the unified `backend.py` abstraction.

### ✨ Features

- Token Embedding: Learnable embedding table mapping token IDs → vectors.
- Positional Embedding: Learnable positional encodings for sequence order.
- Multi‑Head Self‑Attention
    - Q, K, V projections
    - Scaled dot‑product attention
    - Head splitting/merging
    - Output projection
    - Autoregressive causal masking
- Feed‑Forward Network (FFN): Two‑layer MLP with GELU/ReLU activation.
- LayerNorm + Residual Connections: Pre‑norm architecture for stability.
- Stacked Transformer Blocks: Configurable depth, hidden size, number of heads.
- Output Projection: Maps hidden states back to vocabulary logits.
- GPU Acceleration: All operations run on CuPy when available.
- Manual Backpropagation: Full gradient implementation without autograd.

### Tokenizers

SaANN includes simple educational tokenizers:

- `ByteTokenizer`: 256‑token vocabulary
- `CharTokenizer`: ASCII/UTF‑8 character vocabulary

Both provide:

- `encode(text)` → list of token IDs
- `decode(tokens)` → string reconstruction

### Training Utilities

SaANN provides a full training pipeline:

- random sequence batching
- cross‑entropy loss
- AdamW optimizer
- cosine learning‑rate scheduler
- checkpoint saving/loading
- GPU/CPU backend switching

### Text Generation

Autoregressive generation with:

- temperature
- top‑p sampling
- repetition penalty

## 🔀 Cross‑Training MLPs — Experimental Feature

SaANN introduces an experimental training strategy called Cross‑Training, where two identical MLPs are trained in parallel and periodically swap their weights.
This mechanism acts as a form of implicit regularization, encouraging the networks to explore different optimization paths while preventing early overfitting.

Cross‑Training is available through the class:

```python
CrossTrainingSequentialModel
```

and supports all standard SaANN features (batch training, GPU/CPU backend, weight decay, custom losses).

### ✨ Motivation

Traditional MLP training follows a single optimization trajectory.
Cross‑Training instead maintains two synchronized learners, each performing:

- Independent forward/backward pass
- Weight‑swapping pass (during the cross‑training phase)
- Final update (with or without weight decay)

This creates a dynamic where each model periodically inherits the other's representation, forcing both to generalize rather than overfit to local minima.

### ⚙️ How Cross‑Training Works

1. **Two MLPs are constructed** — both networks share the same architecture, initialization, and optimizer.
2. **Training is split into two phases**:
    - **Phase A — Cross‑Training Phase**

        Runs for: $$ {epochs}_{cross} = epoch \times (1−ftr) $$
        where `ftr` is the *fine tuning ratio*.

        During this phase:
        - Weight decay is disabled (wd = 0)
        - Each batch performs:
          - Independent forward/backward pass
          - Weight swap
          - Second forward/backward pass
          - Final update

        This phase encourages exploration and prevents premature convergence.

    - **Phase B — Fine‑Tuning Phase**

        Runs for: $$ {epochs}_{fine} = epoch \times ftr $$

        During this phase:
        - Weight decay is enabled (if provided)
        - No weight swapping
        - Models refine their parameters normally

We tested two configurations:

1. Weight decay applied only during fine‑tuning
2. Weight decay applied during the entire training

Empirically, weight decay only during fine‑tuning produced more stable results.

### 📊 Summary of Experimental Findings

Using the scikit‑learn diabetes dataset, we ran multiple learning rates and three fine‑tuning ratios (0.1, 0.5, 0.7).
Across 3 full runs:

- Cross‑Training consistently achieved R² ≈ 0.50–0.52 for LR ∈ {0.001, 0.0005}
- Performance was comparable to (and sometimes slightly better than) vanilla MLP
- High learning rates (0.01) were unstable — same as standard MLP
- Fine‑tuning ratio 0.5 gave the most consistent results
- Weight decay only during fine‑tuning produced smoother convergence

These results are in line with typical MLP performance on the diabetes dataset (usually R² ≈ 0.45–0.55), confirming that Cross‑Training is functional and stable.

## 💾 Model Management

- `save_model(path)`: Save models in portable format
- `load_model(path)`: Auto-detect and load any SaANN model
- Architecture reconstruction with versioning
- Prevents incompatible model loading

## 📊 Comprehensive Metrics Suite

- **Classification Metrics**: Precision, Recall, F1 (Macro/Micro/Weighted)
- **Advanced Metrics**: Balanced Accuracy, Specificity, Matthews Correlation Coefficient (MCC)
- **Visualization**: Confusion matrices, ROC curves (One-vs-Rest)
- **Reporting**: `Metrics.report()` for full summaries

## 📈 Training & Evaluation

- Real-time training plots and loss tracking
- Automatic train/test split
- Scaling utilities: Z-score, Min-Max, Log, Mean normalization
- Prediction comparisons and scatter plots

---

⬅ [Back to README](../README.md) · Next: [Quick Start](quickstart.md)
