# volatility.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

from abc import ABC, abstractmethod
from .. import backend as BE


class VolatilityEstimator(ABC):
    """
    Base volatility estimator.
    """

    @abstractmethod
    def transform(self, df):
        raise NotImplementedError


class ParkinsonVolatility(VolatilityEstimator):
    """
    Parkinson high-low volatility estimator.
    """

    def transform(self, df):

        high = df["High"].values
        low = df["Low"].values

        return BE.xp.sqrt(
            (1.0 / (4.0 * BE.xp.log(2.0)))
            * (BE.xp.log(high / low) ** 2)
        )