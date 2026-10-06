from .volatility import ParkinsonVolatility
from .features import HARFeatures
from .targets import ForwardVolatilityTarget
from .sequence import SequenceBuilder


class VolatilityDataset:

    def __init__(
        self,
        estimator=None,
        feature_builder=None,
        target_builder=None,
        sequence_builder=None,
        test_size=0.2
    ):

        self.estimator = (
            estimator
            or ParkinsonVolatility()
        )

        self.feature_builder = (
            feature_builder
            or HARFeatures()
        )

        self.target_builder = (
            target_builder
            or ForwardVolatilityTarget()
        )

        self.sequence_builder = (
            sequence_builder
            or SequenceBuilder()
        )

        self.test_size = test_size

    def prepare(self, df):

        volatility = (
            self.estimator.transform(df)
        )

        features = (
            self.feature_builder.transform(
                volatility
            )
        )

        target = (
            self.target_builder.transform(
                volatility
            )
        )

        X, y = (
            self.sequence_builder.build(
                features,
                target
            )
        )

        split_idx = int(
            len(X) *
            (1.0 - self.test_size)
        )

        X_train = X[:split_idx]
        X_test = X[split_idx:]

        y_train = y[:split_idx]
        y_test = y[split_idx:]

        mean = X_train.mean(
            axis=(0, 1),
            keepdims=True
        )

        std = (
            X_train.std(
                axis=(0, 1),
                keepdims=True
            )
            + 1e-8
        )

        X_train = (
            X_train - mean
        ) / std

        X_test = (
            X_test - mean
        ) / std

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )