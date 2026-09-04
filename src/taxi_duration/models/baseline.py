from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class MedianDurationRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):  # noqa: ANN001
        self.median_ = float(np.median(y))
        return self

    def predict(self, X):  # noqa: ANN001
        return np.full(shape=(len(X),), fill_value=self.median_, dtype=float)
