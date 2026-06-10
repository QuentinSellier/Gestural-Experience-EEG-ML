"""Loading and cleaning of the raw participant EEG Excel files."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AnalysisConfig, ParticipantData

# Substrings identifying columns that are not EEG metric signals (time, markers,
# quality flags, interaction phases, mouse coordinates and clicks).
_NON_METRIC_SUBSTRINGS: tuple[str, ...] = (
    "time", "quality", "tutorial", "video", "catalog", "user", "x", "y", "click",
)


def seed_global_rngs(seed: int) -> None:
    """Seed the standard-library and NumPy global RNGs for a reproducible run.

    ``train_test_split`` and ``RandomForestClassifier`` draw from NumPy's global
    RNG when no explicit ``random_state`` is given, so this single call makes the
    whole pipeline deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)


def clean_to_finite_floats(series: pd.Series) -> np.ndarray:
    """Convert a column to a 1D float array, dropping non-numeric and non-finite values.

    ``float(nan)`` succeeds, so non-finite values (dropped-signal samples) are
    filtered explicitly. This mirrors the original notebook's ``int()`` filtering.
    """
    clean_values: list[float] = []
    for value in series:
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(float_value):
            continue
        clean_values.append(float_value)
    return np.array(clean_values)


def load_participant_file(file_path: Path, config: AnalysisConfig) -> list[np.ndarray]:
    """Load and clean one participant's gesture and mouse EEG metrics.

    Returns a list of 8 arrays: indices 0-3 are the gesture metrics, 4-7 the mouse
    metrics, both in the order engagement, memorization, valence, workload. Only the
    shopping phase (``CatalogInteraction == 1``) and good-quality rows
    (``quality == 1``) are kept.
    """
    n_metrics = config.n_variables_per_condition
    metrics: list[np.ndarray] = [np.array([]) for _ in range(2 * n_metrics)]

    for condition_index, tab_name in enumerate(config.tab_names):
        offset = condition_index * n_metrics
        try:
            frame = pd.read_excel(file_path, sheet_name=tab_name, header=2)

            if "CatalogInteraction" in frame.columns:
                frame = frame.loc[frame["CatalogInteraction"] == 1]
            if "quality" in frame.columns:
                frame = frame.loc[frame["quality"] == 1]

            metric_columns = [
                column
                for column in frame.columns
                if not any(token in column.lower() for token in _NON_METRIC_SUBSTRINGS)
            ]
            frame = frame[metric_columns]

            for index, column in enumerate(frame.columns):
                if index >= n_metrics:
                    break
                metrics[offset + index] = clean_to_finite_floats(frame[column])
        except Exception as error:  # noqa: BLE001 - report and continue on bad sheets
            print(f"\nError processing {file_path.name} sheet '{tab_name}': {error}")

    return metrics


def load_all_participants(config: AnalysisConfig) -> ParticipantData:
    """Load and clean every participant file, warning about and skipping missing ones."""
    n_metrics = config.n_variables_per_condition
    data: ParticipantData = [
        [np.array([]) for _ in range(2 * n_metrics)] for _ in range(config.n_participants)
    ]

    for participant_index, filename in enumerate(config.participant_filenames):
        file_path = config.data_dir / filename
        if not file_path.exists():
            print(f"Warning: file {filename} not found. Skipping.")
            continue
        print(
            f"Loading participant {participant_index + 1}/{config.n_participants}",
            end="\r",
            flush=True,
        )
        data[participant_index] = load_participant_file(file_path, config)

    print("\nData loading completed.")
    return data
