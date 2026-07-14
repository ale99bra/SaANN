# Quick Start

## MLP Example (SequentialModel)

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

## RNN Example

```python
from saann.models import RecurrentModel
from saann.metrics import Metrics
import numpy as np

text = "The quick brown fox jumps over the lazy dog"

text = text * 1000

X, y, c2i, i2c, vocab = generate_text_dataset(text, seq_len=43) #create dataset (external function that generate one-hot targets)

model = RecurrentModel(rnn_type='lstm', gpu=False) # rnn_type can be 'lstm', 'gru' or None for vanilla RNN
model.construct(
    input_dim=vocab,
    hidden_dim=128,
    output_dim=vocab,
    learning_rate=3e-3,
    activation_function="softmax",
    act_function_rnn="tanh",
    many_to_one=False,
    normalization=True # applies RMSNorm
)

model.fit(X, y, epochs=200, batch_size=32, loss_function="cross-entropy", graphical=True)

test_text = "The quick brown fox jumps over the lazy dog"

X, y, c2i, i2c, vocab = generate_text_dataset(test_text, seq_len=1)

y_pred = model.predict(X)

ce = cross_entropy(y_true=y, y_pred=y_pred)
print("CE prediction:", ce)

print(f"Text = {test_text}, seq_len = 4, c2i = {c2i}, i2c: {i2c}")
print("y:",y)
print("y_pred:",y_pred)

i = 0
for yi in y:
    print(f"Real: {np.argmax(yi)}, Pred: {np.argmax(y_pred[i])}")
    i += 1

metrics = Metrics(y_test=y, y_pred=y_pred)
metrics.report()
```

## CNN Example — Flower Classification

Classify flowers using the Kaggle flowers dataset (daisy, dandelion, rose, sunflower, tulip):

### Dataset Preparation

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

### Training the CNN

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

## Model Saving & Loading

The following methods are equivalent for all models (apart from `TransformerModel`).

### Save a trained model

```python
model_cnn.save_model("flower_cnn.pkl")
```

### Load and evaluate

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

## Cross-Training Example

Example of the `CrossTrainingSequentialModel` model:

```python
from saann.models import CrossTrainingSequentialModel

model = CrossTrainingSequentialModel(gpu=False)

layers = [
    (X_train.shape[1], 32, "relu", "he"),
    (32, 1, "linear", "he")
]

model.construct(layers, learning_rate=0.001)

y_pred = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=32,
    wd=0.001,                 # applied only during fine-tuning
    loss_function="mse",
    fine_tuning_ratio=0.5,
    graphical=True
)

test_pred = model.predict(X_test)
```

## TransformerModel Example

```python
from saann import backend as BE
from saann.transformer.transformer_model import TransformerModel
from saann.generation import generate_top_p
from saann.training import train_transformer, load_model, create_optimizer
from saann.tokenizer import CharTokenizer, ByteTokenizer

BE.dtype = BE.xp.float16 # Optional - lighter models

# Create the Tokenizer (either Byte or Char)
tokenizer = ByteTokenizer()  # or: tokenizer = CharTokenizer(text)

# Create model
vocab_size = tokenizer.vocab_size
embed_dim = 512
num_heads = 8
ff_hidden_dim = 2048
num_layers = 8
max_seq_len = 512

model = TransformerModel(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_hidden_dim=ff_hidden_dim,
    num_layers=num_layers,
    max_seq_len=max_seq_len,
    learned_positional=True
)

# Create optimizer
optimizer = create_optimizer(model)

# Process data
encoded_text = tokenizer.encode(text) # 'text' is the input data
data = BE.xp.array(encoded_text, dtype=BE.xp.int32)
split = int(0.9 * len(data))
train_data = data[:split]
val_data   = data[split:]

# Train model
train_transformer(
    model=model,
    optimizer=optimizer,
    data=train_data,
    batch_size=32,
    seq_len=128,
    epochs=150,
    checkpoint_every=15,
    checkpoint_dir="checkpoints",
    tokenizer=tokenizer
)

# Load trained model (OPTIONAL)
model, optimizer, tokenizer, scheduler = load_model("checkpoints/checkpoint_final.npz")

# Validate model
start = BE.xp.array([val_data], dtype=BE.xp.int32)
generated = generate_top_p(
    model=model,
    start_tokens=start,
    max_new_tokens=128,
    p=0.9,
    temperature=0.5,
    rep_penalty=1.2
)

decoded_output = tokenizer.decode(generated[0].tolist())
print(f"Decoded output: {decoded_output}")
```

---

⬅ Previous: [Installation](installation.md) · [Back to README](../README.md) · Next: [API Reference](api-reference.md)
