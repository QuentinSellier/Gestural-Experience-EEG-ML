"""Writing analysis results to CSV files in the output directory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .classification import ClassificationResults
from .config import AnalysisConfig


def ensure_output_dir(config: AnalysisConfig) -> None:
    """Create the output directory if it does not already exist."""
    config.output_dir.mkdir(parents=True, exist_ok=True)


def write_metric_table(matrix: np.ndarray, config: AnalysisConfig, filename: str) -> Path:
    """Write a ``(n_participants, 8)`` metric matrix as a labelled CSV.

    The first column is ``participant_id``; the remaining columns carry descriptive
    ``<metric>_gesture`` / ``<metric>_mouse`` headers.
    """
    frame = pd.DataFrame(matrix, columns=config.variable_labels)
    frame.insert(0, "participant_id", config.participant_ids)
    output_path = config.output_dir / filename
    frame.to_csv(output_path, index=False)
    return output_path


def write_classification_report(results: ClassificationResults, config: AnalysisConfig) -> Path:
    """Write the classification metrics, confusion matrix and importances as one CSV."""
    rows: list[tuple[str, float]] = [
        ("accuracy", results.accuracy),
        ("precision", results.precision),
        ("recall", results.recall),
        ("f1_score", results.f1_score),
        ("confusion_true_negative", results.true_negative),
        ("confusion_false_positive", results.false_positive),
        ("confusion_false_negative", results.false_negative),
        ("confusion_true_positive", results.true_positive),
    ]
    rows += [
        (f"importance_{name}", float(value))
        for name, value in zip(results.feature_names, results.feature_importances)
    ]

    frame = pd.DataFrame(rows, columns=["metric", "value"])
    output_path = config.output_dir / "classification_report.csv"
    frame.to_csv(output_path, index=False)
    return output_path
