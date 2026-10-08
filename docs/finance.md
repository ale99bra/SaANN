# Financial Time-Series Forecasting — Experimental Feature

SaANN includes a lightweight `finance` module designed for volatility forecasting, quantitative finance experimentation, and sequence modelling on financial time-series.

The module provides utilities for transforming OHLCV market data into ML ready datasets, reducing boilerplate code required for volatility forecasting workflows.

All computations are backend-agnostic and automatically run on `NumPy` (CPU) or `CuPy` (GPU) through SaANN's unified backend abstraction.

## ✨ Features
- Volatility Estimation
- Built-in volatility estimators:
- Parkinson Volatility (High-Low estimator)

The volatility estimator converts OHLC price data into a realized volatility series suitable for forecasting.

Future releases may include:

- Garman-Klass Volatility
- Rogers-Satchell Volatility
- Yang-Zhang Volatility
- Close-to-Close Volatility
- HAR Feature Engineering

The finance module automatically generates Heterogeneous Auto-Regressive (HAR) features:

- Daily Realized Volatility (RV₁)
- Weekly Realized Volatility (RV₅)
- Monthly Realized Volatility (RV₂₂)

These features are widely used in academic and professional volatility forecasting applications.

### Forward Volatility Targets

SaANN provides automated target creation for volatility forecasting.

Features are constructed from historical data while targets are generated strictly from future observations:

```
Input Window:
[t-lookback+1 ... t]

Forecast Horizon:
[t+1 ... t+horizon]
```

This alignment prevents look-ahead bias and data leakage.

### Sequence Generation

The `SequenceBuilder` automatically converts financial features into tensors compatible with SaANN's RecurrentModel.

Example output shape:

```
(batch_size, sequence_length, num_features)
```
Ready for:

- Vanilla RNN
- GRU
- LSTM

### Dataset Pipeline

The `VolatilityDataset` class provides a complete preprocessing workflow:

```Plain Text
OHLCV DataFrame
↓
Volatility Estimation
↓
HAR Features
↓
Forward Targets
↓
Sequence Construction
↓
Train/Test Split
↓
Feature Scaling
```
Result:

```Python
X_train, X_test, y_train, y_test
```

ready for training.

## 📊 Financial Metrics

The finance module extends SaANN's existing loss functions with forecasting-specific metrics.

# Available metrics:

- RMSPE (Root Mean Squared Percentage Error)
- MAPE (Mean Absolute Percentage Error)
- MASE (Mean Absolute Scaled Error)
- Directional Accuracy
- Horizon Directional Accuracy

The module also provides:

```Python
metrics.volatility_report()
```
which combines standard SaANN losses:

- MSE
- MAE
- R²
- QLIKE

with financial forecasting metrics in a single report.

Example Report
```Python
report = metrics.volatility_report(
    y_test,
    pred,
    baseline = BE.xp.roll(y_test, 1),
    verbose = True,
    graphical = True
)
```
Output:

```Plain Text
MSE
MAE
R²
QLIKE
RMSPE
MAPE
MASE
Directional Accuracy
```
## 🧠 Example Workflow
```Python
from saann.finance.datasets import VolatilityDataset
from saann.finance import metrics
 
dataset = VolatilityDataset()

X_train, X_test, y_train, y_test = dataset.prepare(df)

model = RecurrentModel(
rnn_type="lstm"
)

model.construct(
    input_dim=X_train.shape[2],
    hidden_dim=32,
    output_dim=1,
    learning_rate=1e-3,
    activation_function="softplus",
    init_function="xavier",
    act_function_rnn="tanh",
    many_to_one=True,
    normalization=False
)

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    wd=1e-5,
    loss_function="qlike"
)

pred = model.predict(X_test)

report = metrics.volatility_report(
    y_test,
    pred,
    baseline=np.roll(y_test, 1),
    verbose=True,
    graphical=True
)
```
## ⚠️ Current Scope

The `finance` module focuses on:

- Volatility forecasting
- Financial sequence modelling
- Feature engineering
- Forecast evaluation

The module does not currently provide:

- Portfolio optimisation
- Risk management tools
- Derivatives pricing
- Algorithmic execution systems

Its primary goal is to simplify the preparation and evaluation of financial datasets for SaANN neural networks.