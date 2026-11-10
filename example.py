"""Fit and evaluate transductive COD on breastw."""

from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from cod import COD


def main():
    # Replace this block with your X, evaluation labels y, and nominal column mask.
    # X: (n, m); y: 1=anomaly, 0=normal; nominals: True=categorical, False=numeric.
    with np.load(Path(__file__).resolve().parent / "data" / "breastw.npz") as data:
        X, y, nominals = data["X"], data["y"], data["nominals"]

    # Experiment preprocessing: remove near-constant columns, keeping their types.
    valid = np.std(X, axis=0) > 1e-6
    X, nominals = X[:, valid], nominals[valid]

    # Simulate five known anomalies. For your data, supply only observed labels;
    # all other rows, including hidden anomalies, must remain unlabeled (zero).
    known = np.random.RandomState(2023).choice(np.flatnonzero(y), 5, replace=False)
    train_y = np.zeros(len(X), dtype=np.int8)
    train_y[known] = 1

    model = COD(n_neg=100, batch_size=256, normalize=True).fit(X, train_y, nominals)
    # Without a type mask, use .fit(X, train_y) to infer categorical columns.
    scores = model.predict_score()  # All fitted rows, in order; higher is more anomalous.

    # All rows participate in fitting; only unlabeled rows enter evaluation.
    evaluate = train_y == 0
    auroc = roc_auc_score(y[evaluate], scores[evaluate])
    ap = average_precision_score(y[evaluate], scores[evaluate])
    print("breastw: AUROC={:.8f} AP={:.8f}".format(auroc, ap))


if __name__ == "__main__":
    main()
