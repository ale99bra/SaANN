import unittest
import numpy as np
import pandas as pd

from saann.finance.volatility import ParkinsonVolatility
from saann.finance.features import HARFeatures
from saann.finance.targets import ForwardVolatilityTarget
from saann.finance.sequence import SequenceBuilder
from saann.finance.datasets import VolatilityDataset
from saann.finance import metrics as mtr
from saann.models import RecurrentModel

from saann.finance.metrics import (
    rmspe,
    mape,
    mase,
    directional_accuracy,
    horizon_directional_accuracy,
    volatility_report
)


class TestFinance(unittest.TestCase):
    """
    Test suite for Finance utilities.
    """

    def setUp(self):

        np.random.seed(42)

        n = 250

        close = 100 + np.cumsum(
            np.random.normal(
                0,
                1,
                n
            )
        )

        high = close * (
            1.0
            + np.random.uniform(
                0.0,
                0.03,
                n
            )
        )

        low = close * (
            1.0
            - np.random.uniform(
                0.0,
                0.03,
                n
            )
        )

        volume = np.random.randint(
            1000,
            100000,
            n
        )

        self.df = pd.DataFrame({
            "Close": close,
            "High": high,
            "Low": low,
            "Volume": volume
        })

    def TestVolatilityEstimator(self):

        estimator = ParkinsonVolatility()

        vol = estimator.transform(
            self.df
        )

        if len(vol) != len(self.df):
            raise ValueError(
                "Volatility length mismatch."
            )

        if np.any(vol <= 0):
            raise ValueError(
                "Negative volatility detected."
            )

    def TestHARFeatures(self):

        estimator = ParkinsonVolatility()

        vol = estimator.transform(
            self.df
        )

        har = HARFeatures()

        X = har.transform(vol)

        if X.shape[1] != 3:
            raise ValueError(
                "HAR feature count should be 3."
            )

        if X.shape[0] != len(vol):
            raise ValueError(
                "HAR output length mismatch."
            )

    def TestTargetBuilder(self):

        estimator = ParkinsonVolatility()

        vol = estimator.transform(
            self.df
        )

        target = ForwardVolatilityTarget(
            horizon=5
        )

        y = target.transform(vol)

        if len(y) != len(vol):
            raise ValueError(
                "Target length mismatch."
            )

    def TestSequenceBuilder(self):

        estimator = ParkinsonVolatility()

        vol = estimator.transform(
            self.df
        )

        features = HARFeatures().transform(
            vol
        )

        target = (
            ForwardVolatilityTarget(
                horizon=5
            )
            .transform(vol)
        )

        builder = SequenceBuilder(
            lookback=15
        )

        X, y = builder.build(
            features,
            target
        )

        if X.ndim != 3:
            raise ValueError(
                "Sequence tensor should be 3D."
            )

        if y.ndim != 2:
            raise ValueError(
                "Targets should be 2D."
            )

        if X.shape[1] != 15:
            raise ValueError(
                "Incorrect sequence length."
            )

    def TestDataset(self):

        dataset = VolatilityDataset()

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = dataset.prepare(
            self.df
        )

        if len(X_train) == 0:
            raise ValueError(
                "Empty training set."
            )

        if len(X_test) == 0:
            raise ValueError(
                "Empty test set."
            )

        if X_train.shape[2] != 3:
            raise ValueError(
                "Unexpected feature count."
            )

    def TestMetrics(self):

        y_true = np.array([
            0.10,
            0.15,
            0.20,
            0.18,
            0.25,
            0.30
        ])

        y_pred = np.array([
            0.11,
            0.14,
            0.19,
            0.20,
            0.24,
            0.29
        ])

        score_rmspe = rmspe(
            y_true,
            y_pred
        )

        score_mape = mape(
            y_true,
            y_pred
        )

        score_mase = mase(
            y_true,
            y_pred
        )

        da = directional_accuracy(
            y_true,
            y_pred
        )

        hda = (
            horizon_directional_accuracy(
                y_true,
                y_pred,
                step=2
            )
        )

        if score_rmspe < 0:
            raise ValueError(
                "Invalid RMSPE."
            )

        if score_mape < 0:
            raise ValueError(
                "Invalid MAPE."
            )

        if score_mase < 0:
            raise ValueError(
                "Invalid MASE."
            )

        if not (0 <= da <= 100):
            raise ValueError(
                "Invalid DA."
            )

        if not (0 <= hda <= 100):
            raise ValueError(
                "Invalid Horizon DA."
            )

    def TestVolatilityReport(self):

        y_true = np.array([
            0.10,
            0.20,
            0.15,
            0.30
        ])

        y_pred = np.array([
            0.11,
            0.19,
            0.14,
            0.29
        ])

        report = volatility_report(
            y_true,
            y_pred
        )

        required_keys = [
            "MSE",
            "MAE",
            "R2",
            "QLIKE",
            "RMSPE",
            "MAPE",
            "MASE"
        ]

        for key in required_keys:

            if key not in report:
                raise KeyError(
                    f"{key} missing from report."
                )

    def TestLSTMVolatilityTraining(self):

        dataset = VolatilityDataset()
        X_train, X_test, y_train, y_test = dataset.prepare(self.df)
        model = RecurrentModel(
        rnn_type="lstm"
        )
        model.construct(
            input_dim=X_train.shape[2],
            hidden_dim=4,
            output_dim=1,
            learning_rate=1e-3,
            activation_function="softplus",  # Now operates in active gradient regime
            init_function="xavier",
            act_function_rnn="tanh",
            many_to_one=True,
            normalization=False
        )

        model.fit(
            X_train, y_train,
            epochs=2,
            batch_size=16,
            wd=1e-5,
            loss_function="qlike"
        )

        pred = model.predict(X_test)

        report = mtr.volatility_report(
            y_test,
            pred,
            baseline=y_test,
            verbose=True,
            graphical=True
        )

        assert pred.shape == y_test.shape
        assert "QLIKE" in report

if __name__ == "__main__":
    unittest.main()
