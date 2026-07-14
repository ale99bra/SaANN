# API Documentation

## Models

### SequentialModel (MLP)

Main class for building and training multi-layer perceptrons.

#### Methods

**`__init__(gpu)`**

Initialize a `SequentialModel`.

- `gpu` (bool): Enable GPU acceleration (requires CuPy) - If GPU is not available, SaANN automatically switches to CPU

`SequentialModel` expects the input features (`X`) to be of shape `(batch, features)` and input targets (`y`) to be of shape `(batch, 1)` if regression, or a one-hot encoded array of shape `(batch, classes)` if classification.

**`construct(layers_info, learning_rate, batch_norm, dropout)`**

Builds the network architecture.

- `layers_info` (list): List of tuples `(input_size, neurons, activation, initialization)`
- `learning_rate` (float): Learning rate for optimization
- `batch_norm` (bool): Enables batch normalization
- `dropout` (bool): Enables dropout

- Note:
    - if you are training a regression model, it is recommended to have as the last MLP layer: `(previous_layer_neurons, 1, "linear", "he")`
    - if you are training a classification model, it is recommended to have as the last MLP layer: `(previous_layer_neurons, num_classes, "softmax", "he")` - this will override the selected loss function to `cross-entropy`

**`fit(X, y, epochs, batch_size, wd, loss_function, graphical, real_time, log_plot)`**

Trains the model on data.

- `X` (array): Training features
- `y` (array): Training labels
- `epochs` (int): Number of training epochs
- `batch_size` (int): Batch size for training
- `wd` (float): Weight decay factor for regularization
- `loss_function` (str): 'MSE', 'MAE', 'Cross-entropy' or 'Huber' (or 'Huber:delta' for custom delta)
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
- `loss_function` (str): 'MSE', 'MAE', 'Cross-entropy' or 'Huber' (or 'Huber:delta' for custom delta)
- `split_test_percentage` (float): Test/Train split ratio (0.0-1.0)
- `scaling` (str): Scaling method ('zscore', 'minmax', 'log', 'mean', or None)
- `graphical` (bool): Display training plot
- `real_time` (bool): Update training plot in real-time
- `log_plot` (bool): Display plot in semilogy scale
- `test_loss` (bool): Print loss metrics
- `scatter_comparison` (bool): Display scatter plot of test vs. predicted
- Returns: Tuple of (predictions, train_predictions, X_train, X_test, y_train, y_test)

**`save_model(path)`**

Save the model's architecture and weights in a pickle file
- `path` (str): File path for output

### CNN (Convolutional Neural Network)

Experimental but fully functional convolutional neural network for image classification.

#### Methods

**`__init__(filter_size, num_filters, padding, stride, num_channels, activation_function, init_function, gpu)`**

Initialize a CNN.

- `filter_size` (int): Size of convolution filters - Default = 3
- `num_filters` (int): Number of filters in the first conv layer - Default = 4
- `padding` (int): Padding for convolutions - Default = 1
- `stride` (int): Stride for convolutions - Default = 1
- `num_channels` (int): Number of channels of the input images - Default = 3
- `activation_function` (str): Activation function of ConvBlock - Default = "relu"
- `init_function` (str): Initialization of the ConvBlock weights - Default = "he"
- `gpu` (bool): Enable GPU acceleration (requires CuPy) - If GPU is not available, SaANN automatically switches to CPU

SaANN's CNN expects images in NHWC format `(batch, height, width, channels)`.
The shapes of the input arrays need to follow:
```python
X.shape == (num_samples, height, width, num_channels)
y.shape == (num_samples, num_classes)
```
**Important**: `y` needs to be one-hot encoded - you can use either the `ImageProcessing` class or the `one_hot_vector` function in *saann.processing*.

**`construct(layers_info, learning_rate, batch_norm, dropout)`**

Builds the network architecture.

- `layers_info` (list): List of tuples `(input_size, neurons, activation, initialization)`
- `learning_rate` (float): Learning rate for optimization
- `batch_norm` (bool): Enables batch normalization
- `dropout` (bool): Enables dropout

The last layer should follow `(previous_layer_neurons, num_classes, "softmax", "he")`

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

### RecurrentModel (RNN)

Main class for building and training a Recurrent model.

#### Methods

**`__init__(rnn_type, gpu)`**

Initialize a `RecurrentModel`.

- `rnn_type` (str): Changes architecture of the RNN layer: 'lstm', 'gru' or None (for vanilla)
- `gpu` (bool): Enable GPU acceleration (requires CuPy) - If GPU is not available, SaANN automatically switches to CPU

`RecurrentModel` expects the input features (`X`) to be of shape `(batch, seq_len, features)` and input targets (`y`) to be of shape `(batch, 1)` if regression, or a one-hot encoded array of shape `(batch, classes)` if classification.

**`construct(input_dim, hidden_dim, output_dim, activation_function, init_function, learning_rate, random_scale, act_function_rnn, init_function_rnn, random_scale_rnn, many_to_one, dropout)`**

Construct the RNN and the Dense layers of the model

- `input_dim` (int): Input dimensions
- `hidden_dim` (int): Hidden dimensions
- `output_dim` (int): Output dimensions
- `activation_function` (str): Activation function for the Dense layer
- `init_function` (str): Initiation function for the Dense layer
- `learning_rate` (float): Hyperparameter - Learning rate for optimization
- `random_scale` (float): Scale for the 'random' initiation method (dense layer)
- `act_function_rnn` (str): Activation function for the RNN layer
- `init_function_rnn` (str): Initiation function for the RNN layer
- `random_scale_rnn` (float): Scale for the 'random' initiation method (RNN layer)
- `many_to_one` (bool): Enables the many-to-one or the many-to-many if False
- `dropout` (bool): Enables dropout

**`fit(X, y, epochs, batch_size, wd, loss_function, graphical, real_time, log_plot)`**

Trains the model on data.

- `X` (array): Training features
- `y` (array): Training labels
- `epochs` (int): Number of training epochs
- `batch_size` (int): Batch size for training
- `wd` (float): Weight decay factor for regularization
- `loss_function` (str): 'MSE', 'MAE', 'Cross-entropy' or 'Huber' (or 'Huber:delta' for custom delta)
- `graphical` (bool): Display training plot
- `real_time` (bool): Update training plot in real-time
- `log_plot` (bool): Display plot in semilogy scale
- Returns: Final prediction of training (array)

**`predict(X)`**

Makes predictions on new data.

- `X` (array): Input features
- Returns: Predictions (array)

**`save_model(path)`**

Save the model's architecture and weights in a pickle file
- `path` (str): File path for output

### CrossTrainingSequentialModel

Experimental class implementing dual‑network training with weight‑swapping regularization.

#### Methods

**`__init__(gpu)`**

Initializes the Cross‑Training model.
- `gpu` (bool): Enable GPU acceleration (CuPy). Falls back to CPU automatically.

**`construct(layers_info, learning_rate, batch_norm, dropout)`**

Builds two identical MLPs.

- `layers_info` (list): Same format as `SequentialModel`
- `learning_rate` (float): SGD learning rate
- `batch_norm` (bool): Enable batch normalization. Default is `False`
- `dropout` (bool): Enable dropout. Default is `False`

**`fit(X, y, epochs, batch_size, wd, loss_function, fine_tuning_ratio, graphical, real_time, log_plot, parallelize)`**

Trains the two MLPs using the Cross‑Training strategy.
- `X` (array): Training features
- `y` (array): Training labels
- `epochs` (int): Total number of epochs
- `batch_size` (int): Batch size
- `wd` (float): Weight decay
    - If `wd` > 0, weight decay is applied only during fine‑tuning. Cross‑training phase always uses `wd` = 0.
- `loss_function` (str): 'mse', 'mae', 'huber', 'cross-entropy'
- `fine_tuning_ratio` (float): Fraction of epochs dedicated to fine‑tuning
- `graphical` (bool): Plot training loss. Default is `False`
- `real_time` (bool): Live plot updates. Default is `False`
- `log_plot` (bool): Semilogy scale. Default is `False`
- `parallelize` (bool): Parallel forward/backward passes (CPU only). Default is `False`
- Returns: Final predictions on training data (array)

**`predict(X)`**

Returns predictions using the best-performing model.

- `X` (array): Input features
- Returns: Prediction array

Additional regards:
- If model1 has lower training loss → use model1
- Otherwise → use model2
- If unclear → average predictions

**`save_model(path)`**

Save the model's architecture and weights in a pickle file
- `path` (str): File path for output

### Load Models

**`load_model(path)`**

Load the model saved using the `model.save_model(path)` method. It automatically detects the type of model saved.

- `path` (str): File path of model's pickle
- Returns: Model (class)

## Transformer API
The training of the TransformerModel operates on multiple layers of the architecture. For that reason, it was preferred to provide the API needed for training as in the example found in [Quick Start](docs/quickstart.md#transformermodel-example).

### Tokenizer

For the **tokenizer**, one can use two methods.
From `saann.tokenizer`:

**`ByteTokenizer()`**
- 256‑token vocabulary
- Returns: tokenizer

**`CharTokenizer(text)`**
- `text` (array): Input (text) array
- ASCII/UTF‑8 character vocabulary
- Returns: tokenizer

Both provide:

- **`encode(text)`**
    - returns the list of token IDs
- **`decode(tokens)`**
    - returns the string after reconstruction

To initialize the model, from `saann.transformer.transformer_model`:

### TransformerModel
Class for the transformer model

**`__init__(self, vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_seq_len, learned_positional)__`**

Initialize the model
- `vocab_size` (int): Size of the tokenizer vocabolary
- `embed_dim` (int): Dimensions of the embeddings
- `num_heads` (int): Number of heads
- `ff_hidden_dim` (int): Number of hidden dimension of the Feed-Forward
- `num_layers` (int): Number of layers
- `max_seq_len` (int): Maximum value for the sequence length
- `learned_positional` (bool): tag tokens with unique positions in a sequence

### Training

To train the model, from `saann.training`:

**`create_optimizer(model, learning_rate, wd)`**

Creates the optimizer for the training
- `model` (class): Model class
- `learning_rate` (float): Maximum learning rate for the scheduler. Default is `1e-4`
- `wd` (float): Weight decay factor. Default is `0.1`
- Returns: optimizer class

**`train_transformer(model, optimizer, data, batch_size, seq_len, epochs, checkpoint_every, checkpoint_dir, tokenizer)`**

Operates the pipeline for training the TransformerModel
- `model` (class): Model class
- `optimizer` (class):
- `data` (array): Encoded input array
- `batch_size` (int): Size of batch for training
- `seq_len` (int): Sequence length
- `epochs` (int): Number of epochs
- `checkpoint_every` (int): Number of epochs needed to save each checkpoint
- `checkpoint_dir` (str): File path for the checkpoint directory. Default is "checkpoints". Saves fully trained model as "checkpoint_final.npz"
- `tokenizer` (class): Tokenizer class

#### Load Models

**`load_model(path)`**

Load the model saved during the training.

- `path` (str): File path of the model
- Returns: Model (class)

### Generation

To generate the output, from `saann.generate`:

**`generate_top_p(model, start_tokens, max_new_tokens, p, temperature, rep_penalty)`**

Generate tokens using the trained model
- `model` (class): Trained model class
- `start_tokens` (array): Array of encoded token to initialize the generation
- `max_new_tokens` (int): Number of tokens to generate
- `p` (float): Nucleus sampling threshold
- `temperature` (float): Temperature parameter
- `rep_penalty` (float): Repetition penalty
- Returns: encoded tokens

## Metrics

Comprehensive metrics suite for model evaluation.

### `correlation(X, labels, graphical)`

Calculates and returns the correlation matrix of the features array

- `X` (array): Features array
- `labels` (list): List of features' labels - Default is None
- `graphical` (bool): Plots the correlation matrix as a heatmap - Default is True
- Returns: Correlation matrix (array)

### Metrics

#### Methods

**`__init__(y_pred, y_test)`**

Initialize metrics calculator.

- `y_pred` (array): Predicted labels (probabilities, not logits)
- `y_test` (array): True labels (one-hot encoded - use `saans.processing.one_hot_vector()` if needed)

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

## Processing

**`train_test_split(X, y, split_test_percentage)`**

Splits the dataset provided into Train and Test arrays

- `X` (array): Features
- `y` (array): Targets
- `split_test_percentage` (float): Test/Train split ratio
- Returns: Tuple of (X_train, X_test, y_train, y_test)

**`one_hot_vector(y)`**

Format the `y` array for classification

- `y` (array): Targets
- Returns: Formatted targets (array)

### ImageProcessing

The `ImageProcessing` class prepares image datasets for use with SaANN's CNN and MLP models.

It handles:
- directory scanning
- resizing
- cleaning (RGBA → RGB)
- normalization
- one‑hot encoding
- shuffling
- train/test splitting

All images are returned in NHWC format:

```python
(batch, height, width, channels)
```

#### Methods

**`prepare_images(size, amount)`**

Resizes all images in the dataset.

- `size` (int): output resolution (size × size)
- `amount` (int): optional limit of images per class (can be `None`)

- Creates a temporary folder: `resized/`
- Saves all resized images as JPEG (`.jpg`)
- Preserves class names using the folder structure

Folder structure required:
```bash
dataset/
    class1/
    class2/
    class3/
    class4/
    ...
```

**`image_upload(rel_path)`**

Loads all resized images into memory.

- Reads `.jpg` files from the `resized/` directory
- Extracts class labels from filename prefixes (`class_index.jpg`)
- Returns raw pixel arrays before cleaning

**`clean_dataset(X, y)`**

Ensures all images have valid channels.

- Drops grayscale images (channels < 3)
- Converts RGBA → RGB by removing the alpha channel
- Keeps only 3‑channel images
- Prints whether cleaning was required

**`prepare_features(X=None)`**

Normalizes image pixel values.

- Converts to `float32`
- Normalizes to the range [-0.5, 0.5]
- Converts list of arrays into a single NumPy/CuPy array

**Important**:
Do not normalize manually — SaANN does it automatically.

**`prepare_targets(y=None, list_classes=None)`**

Converts class labels into one‑hot encoded vectors.

- Output shape: `(num_samples, num_classes)`
- Required for all classification tasks
- Class order is alphabetical unless overridden

Example:
```bash
['daisy', 'rose', 'tulip'] → [1,0,0], [0,1,0], [0,0,1]
```

**`shuffle_dataset(X=None, y=None)`**

Randomly shuffles features and labels together.

- Ensures X and y stay aligned
- Uses NumPy/CuPy permutation

**`ready_dataset(size, amount, shuffle, remove_resized, split_test_percentage)`**

Load and prepare an image dataset. It is the full **default** image pre-processing pipeline

- `size` (int): Resize images to size×size
- `amount` (int): Maximum images per class. Can be None.
- `shuffle` (bool): Shuffle the dataset
- `remove_resized` (bool): Remove resized files after processing
- `split_test_percentage` (float): Test/Train split ratio
- Returns: Tuple of `(X_train, X_test, y_train, y_test, class_names)`

Where:

- `X` has shape `(batch, height, width, channels)`
- `y` is one‑hot encoded
- `class_names` is a list of class labels

Pipeline steps:

1. Resize images
2. Load resized images
3. Clean channels
4. Normalize features
5. One‑hot encode labels
6. Shuffle (optional)
7. Train/test split (optional)
8. Delete resized folder (optional)

**Important**:
- SaANN's CNN expects images in NHWC format `(batch, height, width, channels)`
- The CNN model has a parameter `num_channel` that allows to change the number of channels the input images have
- Its default value is 3 but can be changed by the user
- Be aware that if the data was processed using `ready_dataset` then `num_channel` MUST remain 3
- If you change `num_channel`, please provide your own pre-processed data

### Scaling

Data scaling utilities.

#### Methods

**`zScore(x)`, `MinMax(x)`, `LogNorm(x)`, `MeanNorm(x)`**

Apply respective scaling transformations to features.

- `x` (array): Features to scale
- Returns: Scaled features (array)

## Losses

**`MSE(y_true, y_pred)`**

Mean Squared Error loss.

- `y_true` (array): Targets used for testing
- `y_pred` (array): Predicted targets

**`MAE(y_true, y_pred)`**

Mean Absolute Error loss.

- `y_true` (array): Targets used for testing
- `y_pred` (array): Predicted targets

**`cross_entropy(y_true, y_pred)`**

Cross-Entropy Error loss.

- `y_true` (array): Targets used for testing
- `y_pred` (array): Predicted targets

**`Huber(y_true, y_pred, delta)`**

Huber loss with configurable delta parameter.

- `y_true` (array): Targets used for testing
- `y_pred` (array): Predicted targets

**`R2_score(y_true, y_pred)`**

Coefficient of determination (R²).

- `y_true` (array): Targets used for testing
- `y_pred` (array): Predicted targets

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

---

⬅ Previous: [Quick Start](quickstart.md) · [Back to README](../README.md) · Next: [Architecture](architecture.md)
