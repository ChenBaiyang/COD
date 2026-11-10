"""Preprocessing for the fitted COD universe."""

import numpy as np


def _infer_nominals(X):
    """Identify columns with 2–5 distinct, exactly integer-valued entries."""
    mask = np.zeros(X.shape[1], dtype=bool)
    for feature, column in enumerate(X.T):
        integer_valued = column.dtype.kind in "biu" or np.equal(column, np.floor(column)).all()
        if integer_valued:
            mask[feature] = 2 <= np.unique(column).size <= 5
    return mask


def minmax_scale(X, nominals):
    """Copy a float64 matrix, scaling numeric columns; nominals is a bool mask."""
    X = X.copy()
    numerical = ~nominals
    if numerical.any():
        values = X[:, numerical]
        minimum = values.min(axis=0)
        span = np.ptp(values, axis=0)
        span[span == 0] = 1
        X[:, numerical] = (values - minimum) / span
    return X
