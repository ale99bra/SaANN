# losses.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

from . import backend as BE
from . import activation_functions as AF
from functools import partial
import math

def get_loss_functions(loss_spec):
    if not isinstance(loss_spec, str):
        raise TypeError("loss_function must be a string.")

    parts = loss_spec.split(":")
    if len(parts) > 2:
        raise ValueError(
            f"Invalid loss specification {loss_spec!r}; expected 'Huber', 'Huber:delta', or 'QLIKE:eps'."
        )

    name = parts[0].strip().lower()
    has_param = len(parts) == 2

    if name == "huber":
        if has_param:
            try:
                delta = float(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid Huber delta {parts[1]!r}; expected a finite positive number."
                ) from exc
        else:
            delta = 1.0

        if not math.isfinite(delta) or delta <= 0:
            raise ValueError("Huber delta must be a finite positive number.")

        return (
            partial(Huber, delta=delta),
            partial(Huber_der, delta=delta),
            name,
        )

    if name == "qlike":
        if has_param:
            try:
                eps = float(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid QLIKE epsilon {parts[1]!r}; expected a finite positive number."
                ) from exc
        else:
            eps = 1e-8

        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("QLIKE epsilon must be a finite positive number.")

        return (
            partial(QLIKE, eps=eps),
            partial(QLIKE_der, eps=eps),
            name,
        )

    if has_param:
        raise ValueError(f"Only 'Huber' and 'QLIKE' accept parameter suffixes; got {loss_spec!r}.")

    functions = {
        "mse": (MSE, MSE_der),
        "mae": (MAE, MAE_der),
        "cross-entropy": (cross_entropy, cross_entropy_der),
    }

    try:
        loss_func, loss_gradient = functions[name]
    except KeyError as exc:
        raise ValueError(f"Unknown loss function {name!r}.") from exc

    return loss_func, loss_gradient, name

# Loss functions and their derivative
def MSE(y_true, y_pred):
    """
    Calculates the Mean Squared Error.\n
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    """
    return BE.xp.mean((y_true - y_pred)**2)

def MSE_der(y_true, y_pred):
    """
    Derivative of the MSE loss w.r.t. y_pred.\n
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    """
    # return 2 * (y_pred - y_true) / y_true.shape[0] #normalized by the size
    return 2 * (y_pred - y_true) / y_true.size

def MAE(y_true, y_pred):
    """
    Calculates the Mean Absolute Error.\n
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    """
    return BE.xp.mean(BE.xp.abs(y_true - y_pred))

def MAE_der(y_true, y_pred):
    """
    Calculates the Mean Absolute Error's gradient.\n
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    """
    # return BE.xp.sign(y_pred - y_true) / y_true.shape[0]
    return BE.xp.sign(y_pred - y_true) / y_true.size

def R2_score(y_true, y_pred):
    """
    Calculates the R-squared metric.
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    """

    res_SS = BE.xp.sum((y_true - y_pred)**2)
    tot_SS = BE.xp.sum((y_true - BE.xp.mean(y_true))**2)

    return 1 - res_SS/tot_SS

def Huber(y_true, y_pred, delta = 1.0):
    """
    Calculates the Huber loss.\n
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    :param delta: hyperparameter for defining the threshold - quadratic to linear
    """
    error = y_pred - y_true
    abs_error = BE.xp.abs(error)

    elementwise_loss = BE.xp.where(
        abs_error <= delta,
        0.5 * error**2,
        delta * (abs_error - 0.5 * delta),
    )
    return BE.xp.mean(elementwise_loss)

def Huber_der(y_true, y_pred, delta = 1.0):
    """
    Calculates the Huber loss' gradient.\n
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    :param delta: hyperparameter for defining the threshold - quadratic to linear
    """
    """ diff = y_true - y_pred
    quadratic_der = -diff
    linear_der = -delta * BE.xp.sign(diff)
    
    score_der = BE.xp.mean(BE.xp.where(BE.xp.abs(diff) <= delta, quadratic_der, linear_der))

    return score_der """

    error = y_pred - y_true

    elementwise_gradient = BE.xp.where(
        BE.xp.abs(error) <= delta,
        error,
        delta * BE.xp.sign(error),
    )
    return elementwise_gradient / y_true.size

def cross_entropy(y_true, y_pred, epsilon=1e-12):
    """
    Calculates the cross_entropy loss.\n
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    """
    y_pred = BE.xp.clip(y_pred, epsilon, 1. - epsilon)
    #return -BE.xp.sum(y_true * BE.xp.log(y_pred)) / y_true.shape[0]
    return -BE.xp.mean(BE.xp.sum(y_true * BE.xp.log(y_pred), axis=1))

def cross_entropy_der(y_true, y_pred):
    """
    Calculates the cross_entropy loss' gradient.\n
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    """
    # return (y_pred - y_true)
    return (y_pred - y_true) / y_true.shape[0]

def cross_entropy_logits(logits, target_ids):
    """
    Calculates the cross entropy loss for logits.\n
    Parameters
    ----------
    :param logits: Logits in shape (B, L, V)
    :param target_ids: (B, L) integer token IDs
    """
    B, L, V = logits.shape

    # softmax over last dimension
    logits_2d = logits.reshape(B * L, V)
    probs_2d = AF.softmax(logits_2d)  # uses your existing softmax
    probs = probs_2d.reshape(B, L, V)

    # build one-hot targets
    y_true = BE.xp.zeros_like(probs)
    for b in range(B):
        for l in range(L):
            y_true[b, l, target_ids[b, l]] = 1.0

    # compute CE using your existing function
    loss = cross_entropy(
        y_true.reshape(B * L, V),
        probs.reshape(B * L, V)
    )
    return loss


def cross_entropy_logits_der(logits, target_ids):
    """
    Computes gradient wrt logits. Equivalent to softmax(logits) - one_hot(target).\n
    Parameters
    ----------
    :param logits: Logits in shape (B, L, V)
    :param target_ids: (B, L) integer token IDs
    """
    B, L, V = logits.shape

    logits_2d = logits.reshape(B * L, V)
    probs_2d = AF.softmax(logits_2d)
    probs = probs_2d.reshape(B, L, V)

    # build one-hot
    y_true = BE.xp.zeros_like(probs)
    for b in range(B):
        for l in range(L):
            y_true[b, l, target_ids[b, l]] = 1.0

    # derivative wrt logits
    grad = (probs - y_true)
    return grad

def cross_entropy_logits_with_grad(logits, target_ids):
    B, L, V = logits.shape
    logits_2d = logits.reshape(B * L, V)
    target_ids = BE.xp.asarray(target_ids, dtype=BE.xp.int32).reshape(-1)

    rows = BE.xp.arange(B * L)

    max_logits = BE.xp.max(logits_2d, axis=1, keepdims=True)
    shifted = logits_2d - max_logits
    exp_logits = BE.xp.exp(shifted)
    sum_exp = BE.xp.sum(exp_logits, axis=1, keepdims=True)

    log_probs = shifted - BE.xp.log(sum_exp)
    loss = -BE.xp.mean(log_probs[rows, target_ids])

    probs = exp_logits / sum_exp
    probs[rows, target_ids] -= 1.0

    # Match the mean used by the loss.
    grad_logits = probs / (B * L)

    return loss, grad_logits.reshape(B, L, V)

def QLIKE(y_true, y_pred, eps=1e-8):
    """
    Calculates the Quasi-Likelihood (QLIKE) Loss.
    
    Formula: L(y, y_hat) = (y / y_hat) - ln(y / y_hat) - 1
    
    Parameters
    ----------
    :param y_true: Testing values array (Actual Volatility Variance or StDev).
    :param y_pred: Array of values predicted by the model.
    :param eps: Epsilon clipping threshold for numerical stability.
    """
    # Clip predictions and targets to prevent log(0) or division by zero
    y_pred_safe = BE.xp.clip(y_pred, eps, None)
    y_true_safe = BE.xp.clip(y_true, eps, None)
    
    ratio = y_true_safe / y_pred_safe
    return BE.xp.mean(ratio - BE.xp.log(ratio) - 1.0)


def QLIKE_der(y_true, y_pred, eps=1e-8):
    """
    Derivative of the QLIKE loss w.r.t. y_pred.
    
    Formula: dL/dy_hat = (y_hat - y) / (y_hat^2)
    Normalized across total elements (y_true.size).
    
    Parameters
    ----------
    :param y_true: Testing values array.
    :param y_pred: Array of values predicted by the model.
    :param eps: Epsilon clipping threshold for numerical stability.
    """
    y_pred_safe = BE.xp.clip(y_pred, eps, None)
    y_true_safe = BE.xp.clip(y_true, eps, None)
    
    # Gradient w.r.t predictions
    grad = (y_pred_safe - y_true_safe) / (y_pred_safe ** 2)
    
    # Normalize by total number of elements matching SaANN's convention
    return grad / y_true.size

if __name__ == "__main__":
    pred = BE.xp.linspace(0, 100, num = 26)
    true = BE.xp.linspace(0, 90, num = 26)

    r2_scoring = R2_score(true, pred)
    huber = Huber(true, pred, delta = 0.1)
    huber_der = Huber_der(true, pred, delta = 0.1)
    print(r2_scoring)
    print(huber)
    print(huber_der)