from .. import backend as BE
from ..losses import (
    MSE,
    MAE,
    R2_score,
    QLIKE
)

_EPS = 1e-10


def _flatten(x):
    return BE.xp.asarray(x).reshape(-1)

def rmspe(y_true, y_pred):
    """
    Root Mean Squared Percentage Error.

    Returns percentage.
    """

    y_true = _flatten(y_true)
    y_pred = _flatten(y_pred)

    err = (
        (y_true - y_pred)
        / (y_true + _EPS)
    )

    return float(
        100.0 *
        BE.xp.sqrt(
            BE.xp.mean(err ** 2)
        )
    )

def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error.

    Returns percentage.
    """

    y_true = _flatten(y_true)
    y_pred = _flatten(y_pred)

    return float(
        100.0 *
        BE.xp.mean(
            BE.xp.abs(
                (y_true - y_pred)
                / (y_true + _EPS)
            )
        )
    )

def directional_accuracy(
    y_true,
    y_pred,
    baseline=None
):
    """
    Directional accuracy.

    If baseline is provided:

        y_true > baseline
        y_pred > baseline

    Otherwise:

        sign(diff(y_true))
        sign(diff(y_pred))
    """

    y_true = _flatten(y_true)
    y_pred = _flatten(y_pred)

    if baseline is not None:

        baseline = _flatten(baseline)

        actual = (
            y_true > baseline
        )

        pred = (
            y_pred > baseline
        )

    else:

        actual = (
            BE.xp.diff(y_true) > 0
        )

        pred = (
            BE.xp.diff(y_pred) > 0
        )

    return float(
        100.0 *
        BE.xp.mean(actual == pred)
    )

def horizon_directional_accuracy(
    y_true,
    y_pred,
    step=5
):
    """
    Non-overlapping directional accuracy.
    """

    y_true = _flatten(y_true)
    y_pred = _flatten(y_pred)

    actual = (
        BE.xp.diff(
            y_true[::step]
        ) > 0
    )

    pred = (
        BE.xp.diff(
            y_pred[::step]
        ) > 0
    )

    return float(
        100.0 *
        BE.xp.mean(actual == pred)
    )

def mase(
    y_true,
    y_pred
):
    """
    Mean Absolute Scaled Error.

    Uses naive lag-1 forecast benchmark.
    """

    y_true = _flatten(y_true)
    y_pred = _flatten(y_pred)

    model_mae = (
        BE.xp.mean(
            BE.xp.abs(
                y_true - y_pred
            )
        )
    )

    naive_mae = (
        BE.xp.mean(
            BE.xp.abs(
                y_true[1:]
                - y_true[:-1]
            )
        )
    )

    return float(
        model_mae /
        (naive_mae + _EPS)
    )

def volatility_report(
    y_true,
    y_pred,
    baseline=None,
    horizon_step=None
):
    """
    Standard volatility forecasting report.
    """

    report = {
        "MSE":
            float(
                MSE(y_true, y_pred)
            ),

        "MAE":
            float(
                MAE(y_true, y_pred)
            ),

        "R2":
            float(
                R2_score(
                    y_true,
                    y_pred
                )
            ),

        "QLIKE":
            float(
                QLIKE(
                    y_true,
                    y_pred
                )
            ),

        "RMSPE":
            rmspe(
                y_true,
                y_pred
            ),

        "MAPE":
            mape(
                y_true,
                y_pred
            ),

        "MASE":
            mase(
                y_true,
                y_pred
            )
    }

    if baseline is not None:

        report[
            "Directional Accuracy"
        ] = directional_accuracy(
            y_true,
            y_pred,
            baseline
        )

    if horizon_step is not None:

        report[
            "Horizon Directional Accuracy"
        ] = horizon_directional_accuracy(
            y_true,
            y_pred,
            horizon_step
        )

    return report