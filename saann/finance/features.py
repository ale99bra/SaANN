# features.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

import pandas as pd
from .. import backend as BE


class HARFeatures:
    """
    Heterogeneous AutoRegressive volatility features.
    """

    def __init__(
        self,
        daily_window=1,
        weekly_window=5,
        monthly_window=22
    ):
        self.daily_window = daily_window
        self.weekly_window = weekly_window
        self.monthly_window = monthly_window

    def transform(self, volatility):

        vol = pd.Series(volatility)

        rv_daily = (
            vol.rolling(
                self.daily_window,
                min_periods=1
            )
            .mean()
            .values
        )

        rv_weekly = (
            vol.rolling(
                self.weekly_window,
                min_periods=1
            )
            .mean()
            .values
        )

        rv_monthly = (
            vol.rolling(
                self.monthly_window,
                min_periods=1
            )
            .mean()
            .values
        )

        return BE.xp.column_stack(
            [
                rv_daily,
                rv_weekly,
                rv_monthly
            ]
        )