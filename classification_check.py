"""Classification check reported in the article (Section 5.2 and Appendix A.4).

Reads the per-participant EEG summaries written by ``recalcul_tableaux_eeg.py``
(``output/eeg_comparisons.xlsx``, sheets "Peaks" and "Proportions") and:

- ``main``: estimates the accuracy of a Random Forest (100 trees, default
  hyperparameters) trained on the eight features (max peak and proportion above
  threshold of the four metrics), over participant-grouped splits (20% of the
  participants in the test set). Reports the mean accuracy, its variability across
  splits, the confusion matrix and, for reference, a model restricted to the two
  valence (attraction) features.
- ``perm``: permutation test that preserves the participant pairing. In each
  permutation, the gesture and mouse labels are swapped at random within each
  participant (each participant keeps one session of each label), and the accuracy is
  re-estimated on the same kind of splits. p = (1 + #null >= observed) / (1 + #perms).

Usage::

    uv run python classification_check.py main [--splits 10000]
    uv run python classification_check.py perm [--perms 200 --splits-per-perm 100]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

# Column order of the summary sheets: 4 gesture metrics, then the same 4 for mouse.
NAMES = ["engagement", "memorization", "valence", "workload"]


def load_features(xlsx: Path, cols: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build X (one row per session), y (gesture=1, mouse=0) and participant groups."""
    peak = pd.read_excel(xlsx, "Peaks").iloc[:, 1:].to_numpy()
    prop = pd.read_excel(xlsx, "Proportions").iloc[:, 1:].to_numpy()
    rows, labels = [], []
    for i in range(peak.shape[0]):
        rows.append([peak[i, c] for c in cols] + [prop[i, c] for c in cols])
        labels.append(1)
        rows.append([peak[i, c + 4] for c in cols] + [prop[i, c + 4] for c in cols])
        labels.append(0)
    names = [f"max_peak_{NAMES[c]}" for c in cols] + [f"prop_above_threshold_{NAMES[c]}" for c in cols]
    groups = np.repeat(np.arange(peak.shape[0]), 2)
    return np.array(rows), np.array(labels), groups, names


def one_split(X, y, groups, seed):
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, groups))
    clf = RandomForestClassifier(n_estimators=100, random_state=seed + 1)
    clf.fit(X[train], y[train])
    pred = clf.predict(X[test])
    return (pred == y[test]).mean(), confusion_matrix(y[test], pred, labels=[0, 1]).ravel()


def estimate(X, y, groups, n_splits, seed):
    """Mean accuracy and confusion shares over ``n_splits`` participant-grouped splits."""
    rng = np.random.default_rng(seed)
    acc, conf = [], np.zeros(4)
    for _ in range(n_splits):
        a, c = one_split(X, y, groups, int(rng.integers(2**31 - 1)))
        acc.append(a)
        conf += c
    acc = np.array(acc)
    return {
        "splits": n_splits,
        "accuracy_mean": float(acc.mean()),
        "accuracy_sd_between_splits": float(acc.std(ddof=1)) if n_splits > 1 else 0.0,
        "accuracy_p2.5": float(np.percentile(acc, 2.5)),
        "accuracy_p97.5": float(np.percentile(acc, 97.5)),
        "confusion_share_tn_fp_fn_tp": (conf / conf.sum()).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["main", "perm"])
    ap.add_argument("--xlsx", type=Path, default=Path("output") / "eeg_comparisons.xlsx")
    ap.add_argument("--splits", type=int, default=10000)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--splits-per-perm", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("output"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    X, y, groups, names = load_features(args.xlsx, [0, 1, 2, 3])
    if args.mode == "main":
        result = {"features": names, "eight_feature_model": estimate(X, y, groups, args.splits, seed=42)}
        Xv, yv, gv, nv = load_features(args.xlsx, [2])
        result["valence_only"] = {"features": nv, **estimate(Xv, yv, gv, args.splits // 2, seed=43)}
        (args.out / "classification_check.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result["eight_feature_model"].items()}, indent=2))
    else:
        observed = estimate(X, y, groups, args.splits_per_perm, seed=44)["accuracy_mean"]
        rng = np.random.default_rng(45)
        null = []
        for k in range(args.perms):
            swap = rng.integers(0, 2, size=len(np.unique(groups))).astype(bool)
            yp = y.copy()
            for g in np.where(swap)[0]:
                yp[2 * g], yp[2 * g + 1] = y[2 * g + 1], y[2 * g]
            null.append(estimate(X, yp, groups, args.splits_per_perm, seed=1000 + k)["accuracy_mean"])
            print(f"permutation {k + 1}/{args.perms}", end="\r", flush=True)
        null = np.array(null)
        p = (1 + (null >= observed).sum()) / (1 + len(null))
        result = {
            "splits_per_estimate": args.splits_per_perm,
            "permutations": args.perms,
            "observed_accuracy": float(observed),
            "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)),
            "null_p95": float(np.percentile(null, 95)),
            "p_value": float(p),
            "null_distribution": null.tolist(),
        }
        (args.out / "permutation_test.json").write_text(json.dumps(result, indent=2))
        print(f"\nobserved={observed:.4f} null_mean={null.mean():.4f} p={p:.3f}")


if __name__ == "__main__":
    main()
