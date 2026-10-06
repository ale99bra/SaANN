from .. import backend as BE


class ForwardVolatilityTarget:
    """
    Forward realised volatility target.
    """

    def __init__(
        self,
        horizon=5,
        aggregation="mean"
    ):
        self.horizon = horizon
        self.aggregation = aggregation

    def transform(self, volatility):

        n = len(volatility)
        target = BE.xp.full(n, BE.xp.nan)

        for t in range(n - self.horizon):

            future_window = volatility[
                t + 1 : t + 1 + self.horizon
            ]

            if self.aggregation == "mean":
                target[t] = BE.xp.mean(future_window)

            elif self.aggregation == "max":
                target[t] = BE.xp.max(future_window)

            elif self.aggregation == "median":
                target[t] = BE.xp.median(future_window)

            else:
                raise ValueError(
                    f"Unknown aggregation: {self.aggregation}"
                )

        return target