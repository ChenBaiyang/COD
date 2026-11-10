"""Consistency-guided outlier detection, following Eqs. (5)-(14) of COD."""

import numpy as np

from .preprocessing import _infer_nominals, minmax_scale


RADIUS_GRID = np.arange(1, 1000, dtype=np.float64) / 1000


def _cardinality_curve(values, indices):
    """Mean normalized cardinality at each radius (Eq. 6)."""
    distances = np.abs(values[indices, None] - values[None, :]).ravel()
    distances.sort()
    cumulative = np.concatenate(([0.0], np.cumsum(1 - distances)))
    stops = np.searchsorted(distances, RADIUS_GRID, side="right")
    return cumulative[stops] / len(distances)


def _learn_radius(values, positive, negative):
    objective = (_cardinality_curve(values, negative)
                 - _cardinality_curve(values, positive))
    # Ascending grid and first argmax resolve ties toward smaller radii.
    return float(RADIUS_GRID[np.argmax(objective)])


def _relation(left, right, nominal, radius):
    if nominal:
        return (left[:, None] == right[None, :]).astype(np.float64)
    distances = np.abs(left[:, None] - right[None, :])
    return np.where(distances <= radius, 1 - distances, 0.0)


class COD:
    """Transductive COD; larger scores indicate stronger outlier evidence.

    Numerical features are min-max normalized over the fitted universe.
    Nominal features are equality-coded numbers and are not scaled. Computation
    uses CPU NumPy float64. batch_size limits relation rows, not the universe.
    """

    def __init__(self, n_neg=100, batch_size=256, normalize=True):
        self.n_neg = n_neg
        self.batch_size = batch_size
        self.normalize = normalize
        self._fitted = False

    def _blocks(self, n):
        for start in range(0, n, self.batch_size):
            yield slice(start, min(start + self.batch_size, n))

    def _negative_indices(self, values, nominal, unlabeled):
        n = len(values)
        distances = np.empty(n, dtype=np.float64)
        for block in self._blocks(n):
            if nominal:
                matrix = values[block, None] != values[None, :]
            else:
                matrix = np.abs(values[block, None] - values[None, :])
            distances[block] = matrix.mean(axis=1)
        # unlabeled is in row order; stable sorting preserves the tie order.
        order = np.argsort(distances[unlabeled], kind="stable")
        return unlabeled[order[:self.n_neg]]

    def fit(self, X, train_y, nominals=None):
        """Fit the entire universe with 1=known outlier, 0=unlabeled.

        X accepts (n, m) or (n, m, 1); nominals is a Boolean/0-1 column mask.
        If omitted, columns with 2–5 distinct integer values are categorical.
        Types are determined before normalization; explicit masks take priority.
        """
        self._fitted = False
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               or v <= 0 for v in (self.n_neg, self.batch_size)):
            raise ValueError("n_neg and batch_size must be positive integers")
        if not isinstance(self.normalize, (bool, np.bool_)):
            raise ValueError("normalize must be Boolean")
        if np.iscomplexobj(X):
            raise ValueError("X must contain real values")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 3 and X.shape[-1] == 1:
            X = X[..., 0]
        if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] == 0:
            raise ValueError("X must have shape (n>=2, m>=1) or (n, m, 1)")
        if not np.isfinite(X).all():
            raise ValueError("X must contain finite values")
        n, m = X.shape
        nominal = _infer_nominals(X) if nominals is None else np.asarray(nominals)
        if nominal.shape != (m,) or not np.isin(nominal, [0, 1]).all():
            raise ValueError("nominals must be a Boolean/0-1 mask over columns")
        nominal = nominal.astype(bool)
        if self.normalize:
            X = minmax_scale(X, nominal)
        numerical = X[:, ~nominal]
        if numerical.size and ((numerical < -1e-12).any()
                               or (numerical > 1 + 1e-12).any()):
            raise ValueError("Numerical features must be normalized to [0, 1]")
        labels = np.asarray(train_y)
        if labels.shape != (n,) or not np.isin(labels, [0, 1]).all():
            raise ValueError("train_y must be a 1D 0-1 known-outlier mask")
        positive = np.flatnonzero(labels == 1)
        unlabeled = np.flatnonzero(labels == 0)
        if len(positive) == 0 or len(unlabeled) == 0:
            raise ValueError("At least one known outlier and one unlabeled row are required")

        self.radii_ = np.full(m, -1.0)  # Nominal equality has no learned radius.
        self.consistencies_ = np.empty(m)
        cardinalities = np.empty((m, n))  # Normalized by the full |U|.
        for feature in range(m):
            values = X[:, feature]
            negative = self._negative_indices(values, nominal[feature], unlabeled)
            if not nominal[feature]:
                self.radii_[feature] = _learn_radius(values, positive, negative)
            radius = self.radii_[feature]
            # For crisp classes, Eq. (2) reduces to the minimum complement
            # of the relation to the opposite class.
            cross = _relation(values[positive], values[negative], nominal[feature], radius)
            complements = 1 - cross
            self.consistencies_[feature] = (complements.min(axis=1).mean()
                                            + complements.min(axis=0).mean())
            for block in self._blocks(n):
                relation = _relation(values[block], values, nominal[feature], radius)
                cardinalities[feature, block] = relation.mean(axis=1)
        self.decision_scores_ = ((1 - cardinalities)
                                 * self.consistencies_[:, None]).mean(axis=0)
        self.nominals_ = nominal.copy()
        self._fitted = True
        return self

    def predict_score(self):
        """Return a copy of scores for the fitted universe, in input row order."""
        if not self._fitted:
            raise RuntimeError("Call fit before predict_score")
        return self.decision_scores_.copy()
