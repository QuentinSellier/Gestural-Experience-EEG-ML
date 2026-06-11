"""Configuration constants and shared types for the analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Participant-indexed nested structure: ``data[participant][variable] -> 1D array``.
# Variables 0-3 are the gesture metrics and 4-7 the matching mouse metrics, both in
# the order: engagement, memorization, valence, workload.
ParticipantData = list[list[np.ndarray]]

# The four EEG metrics, in their canonical column order.
METRIC_NAMES: tuple[str, ...] = ("engagement", "memorization", "valence", "workload")


@dataclass(frozen=True)
class AnalysisConfig:
    """Immutable configuration for the full analysis pipeline.

    Grouping every constant here keeps the pipeline functions free of module-level
    globals and makes a run fully described by a single object.
    """

    belgian_base_id: int = 1000
    n_participants: int = 39
    n_variables_per_condition: int = 4
    random_seed: int = 42
    simulations_peak: int = 2000
    simulations_ml: int = 1000
    high_activity_threshold: float = 100.0
    iqr_multiplier: float = 3.0
    rf_n_estimators: int = 100
    test_size: float = 0.2
    # Train/test splitting strategy:
    #   "random"  -> plain row-wise random split (the original paper / notebook). A
    #                participant's gesture and mouse rows may fall on opposite sides.
    #   "grouped" -> participant-level split (GroupShuffleSplit on participant id):
    #                both rows of a participant always stay together, preventing
    #                cross-condition leakage within a participant.
    split_strategy: str = "random"
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    # Sheet names carry a double space, matching the raw Bitbrain export.
    tab_names: tuple[str, str] = ("t01  Test_gestures", "t02  Test_mouse")
    # Features used by the classifier (column indices into a 4-metric block).
    peak_feature_cols: tuple[int, int] = (0, 2)  # max peak: engagement, valence
    prop_feature_cols: tuple[int, int] = (1, 3)  # proportion>threshold: memorization, workload

    @property
    def participant_ids(self) -> list[int]:
        """Return the participant identifiers, e.g. ``[1001, ..., 1039]``."""
        return [self.belgian_base_id + i for i in range(1, self.n_participants + 1)]

    @property
    def participant_filenames(self) -> list[str]:
        """Return the input Excel filename for each participant."""
        return [f"individual_upv_{pid}.xlsx" for pid in self.participant_ids]

    @property
    def variable_labels(self) -> list[str]:
        """Return the 8 column labels: 4 gesture metrics then 4 mouse metrics."""
        gesture = [f"{name}_gesture" for name in METRIC_NAMES]
        mouse = [f"{name}_mouse" for name in METRIC_NAMES]
        return gesture + mouse
