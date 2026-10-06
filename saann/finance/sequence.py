from .. import backend as BE

class SequenceBuilder:

    def __init__(
        self,
        lookback=15
    ):
        self.lookback = lookback

    def build(
        self,
        features,
        target
    ):

        X = []
        y = []

        start = self.lookback - 1

        for end_idx in range(start, len(features)):

            if BE.xp.isnan(target[end_idx]):
                continue

            begin_idx = (
                end_idx
                - self.lookback
                + 1
            )

            X.append(
                features[
                    begin_idx:end_idx + 1
                ]
            )

            y.append(
                target[end_idx]
            )

        return (
            BE.xp.asarray(X),
            BE.xp.asarray(y).reshape(-1, 1)
        )