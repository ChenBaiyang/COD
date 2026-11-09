"""Run COD experiments on the bundled datasets."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import COD


ROOT = Path(__file__).resolve().parent


def load_dataset(path):
    """Load features, evaluation labels, and attribute types."""
    with np.load(path) as archive:
        X, y, nominal = archive["X"], archive["y"], archive["nominals"]
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("Dataset X must be a finite 2D array")
    if y.shape != (len(X),) or not np.isin(y, [0, 1]).all():
        raise ValueError("Dataset y must be a binary vector")
    valid = np.std(X, axis=0) > 1e-6
    return X[:, valid], y.astype(bool), nominal[valid]


def select_known_outliers(y, seed, n_labeled_outliers=5):
    """Simulate known labels and exclude them from evaluation."""
    positives = np.flatnonzero(y)
    if not 0 < n_labeled_outliers < len(positives) or y.all():
        raise ValueError("Need normal rows and more anomalies than the labeled count")
    known = np.random.RandomState(seed).choice(positives, n_labeled_outliers, replace=False)
    train_y = np.zeros(len(y), dtype=np.int8)
    train_y[known] = 1
    return known, train_y, train_y == 0


def run_experiments(datasets=None, seeds=tuple(range(2023, 2033)), output_dir=None,
                    n_labeled_outliers=5, n_neg=100, batch_size=256, data_dir=ROOT / "data",
                    normalize=True):
    """Write per-seed metrics and their summary to a new output directory."""
    available = {p.stem.lower(): p for p in Path(data_dir).glob("*.npz")}
    keys = sorted(available) if datasets is None else [Path(d).stem.lower() for d in datasets]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Select at least one dataset, without duplicates")
    paths = [available[key] for key in keys]
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least one seed, without duplicates")
    if output_dir is None:
        output_dir = ROOT / "outputs" / datetime.now().strftime("cod-%Y%m%d-%H%M%S-%f")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    fields = ["dataset", "model", "seed", "n_labeled_outliers", "n_neg", "n_test",
              "n_features", "auroc", "ap", "elapsed_seconds", "known_outlier_indices"]
    summary = []
    print("Output: " + str(output_dir), flush=True)
    with (output_dir / "results_COD.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in paths:
            X, y, nominal = load_dataset(path)
            metrics = []
            for seed in seeds:
                tick = time.perf_counter()
                known, train_y, test = select_known_outliers(y, seed, n_labeled_outliers)
                model = COD(n_neg=n_neg, batch_size=batch_size, normalize=normalize).fit(X, train_y, nominal)
                scores = model.predict_score()
                auroc = float(roc_auc_score(y[test], scores[test]))
                ap = float(average_precision_score(y[test], scores[test]))
                writer.writerow(dict(dataset=path.stem, model="COD", seed=seed,
                                     n_labeled_outliers=n_labeled_outliers,
                                     n_neg=min(n_neg, len(X) - len(known)), n_test=int(test.sum()),
                                     n_features=X.shape[1], auroc=auroc, ap=ap,
                                     elapsed_seconds=time.perf_counter() - tick,
                                     known_outlier_indices=json.dumps(known.tolist())))
                handle.flush()
                metrics.append((auroc, ap))
                print("{} seed={} AUROC={:.8f} AP={:.8f}".format(path.stem, seed, auroc, ap),
                      flush=True)
            metrics = np.asarray(metrics)
            summary.append(dict(dataset=path.stem, n_runs=len(seeds),
                                auroc_mean=float(metrics[:, 0].mean()),
                                auroc_std=float(metrics[:, 0].std()),
                                ap_mean=float(metrics[:, 1].mean()),
                                ap_std=float(metrics[:, 1].std())))
    summary.append(dict(dataset="Average", n_runs=len(paths) * len(seeds),
                        auroc_mean=float(np.mean([r["auroc_mean"] for r in summary])), auroc_std="",
                        ap_mean=float(np.mean([r["ap_mean"] for r in summary])), ap_std=""))
    with (output_dir / "summary_COD.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", help="Dataset names; default: all bundled datasets")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2023, 2033)))
    parser.add_argument("--output-dir", type=Path, help="New output directory")
    parser.add_argument("--n-labeled-outliers", type=int, default=5)
    parser.add_argument("--n-neg", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false",
                        help="Use numerical inputs already normalized to [0, 1]")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    run_experiments(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
