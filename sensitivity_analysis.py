"""Sensitivity of the EEG comparisons to the outlier boundary (Appendix C, Table C.10).

Recomputes the duration-corrected max peak and the proportion above threshold with
the per-participant IQR boundary Q1 - k*IQR / Q3 + k*IQR for k = 1.5, 3 (main
analysis), 5 and 10, and without outlier treatment. Out-of-range values are replaced
by the boundary in both conditions. For each setting, gesture and mouse are compared
with the one-tailed Wilcoxon signed-rank test used in the article.

Usage::

    uv run python sensitivity_analysis.py [--raw data] [--out output]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from eeg_gesture_classifier.config import AnalysisConfig
from eeg_gesture_classifier.data_loading import load_all_participants, seed_global_rngs
from eeg_gesture_classifier.features import (
    compute_average_corrected_max_peak,
    compute_proportion_above_threshold,
)
from eeg_gesture_classifier.outliers import apply_iqr_filter
from recalcul_tableaux_eeg import ROWS, wilcoxon_one_tailed

# (label, multiplier); None means no outlier treatment.
BOUNDS = [("1.5", 1.5), ("3", 3.0), ("5", 5.0), ("10", 10.0), ("none", None)]


def features_for_bound(raw_dir: Path, k: float | None):
    config = replace(AnalysisConfig(), data_dir=raw_dir, iqr_multiplier=(k if k is not None else 1e12))
    seed_global_rngs(config.random_seed)
    raw = load_all_participants(config)
    filtered = apply_iqr_filter(raw, config)
    prop = compute_proportion_above_threshold(filtered, config)
    seed_global_rngs(config.random_seed)
    peak = compute_average_corrected_max_peak(filtered, config)
    return peak, prop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("output"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    features = {label: features_for_bound(args.raw, k) for label, k in BOUNDS}
    rows = []
    for col, name in ROWS:
        for measure, index in (("Peak", 0), ("Proportion", 1)):
            row = {"Indicator": name, "Measure": measure}
            for label, _ in BOUNDS:
                values = features[label][index]
                z, p = wilcoxon_one_tailed(values[:, col], values[:, col + 4])
                row[f"Z_k{label}"] = z
                row[f"p_k{label}"] = p
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "sensitivity_outlier_boundary.csv", index=False)
    print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
