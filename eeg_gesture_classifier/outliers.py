"""Outlier handling for the EEG metrics."""

from __future__ import annotations

import numpy as np

from .config import AnalysisConfig, ParticipantData


def apply_iqr_filter(data: ParticipantData, config: AnalysisConfig) -> ParticipantData:
    """Apply a per-participant IQR filter (outer boundary, multiplier 3).

    The acceptance range is computed on the pooled gesture+mouse signal for each
    participant and metric. In both conditions, out-of-range values are replaced by
    the boundary value, so that session lengths are preserved.
    """
    n_participants = config.n_participants
    n_metrics = config.n_variables_per_condition
    filtered: ParticipantData = [
        [np.array([]) for _ in range(2 * n_metrics)] for _ in range(n_participants)
    ]

    for metric_index in range(n_metrics):
        for participant_index in range(n_participants):
            gesture = np.asarray(data[participant_index][metric_index], dtype=float)
            mouse = np.asarray(data[participant_index][metric_index + n_metrics], dtype=float)

            combined = np.concatenate((gesture, mouse), axis=None)
            if len(combined) == 0:
                continue

            q1 = np.percentile(combined, 25)
            q3 = np.percentile(combined, 75)
            iqr = q3 - q1
            lower_bound = q1 - config.iqr_multiplier * iqr
            upper_bound = q3 + config.iqr_multiplier * iqr

            filtered[participant_index][metric_index] = np.clip(gesture, lower_bound, upper_bound)
            filtered[participant_index][metric_index + n_metrics] = np.clip(
                mouse, lower_bound, upper_bound
            )

    return filtered
