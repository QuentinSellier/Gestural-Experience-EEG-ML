"""Outlier handling for the EEG metrics.

Correction of 23-11-2026
------------------------
The method described in the article (Appendix A.2) and in the doctoral thesis
(Essay 3) replaces every out-of-range value by the boundary value, in both
interaction conditions. The original notebooks (January-February 2023) applied
this replacement to the gesture session only, and deleted the out-of-range
values of the mouse session. This asymmetry is not described anywhere and
creates, by itself, a difference in favour of the gesture condition (the
gesture peak equals the upper boundary whenever a value exceeds it, while the
mouse peak is necessarily lower).

``apply_iqr_filter`` implements the published method and is used by the
corrected pipeline. ``apply_original_asymmetric_filter`` reproduces the
original notebook logic and is kept only to document and compare the
published results. It must not be used for new analyses.
"""

from __future__ import annotations

import numpy as np

from .config import AnalysisConfig, ParticipantData


def _bounds(gesture: np.ndarray, mouse: np.ndarray, config: AnalysisConfig):
    """IQR bounds (outer boundary) computed on the pooled gesture+mouse signal."""
    combined = np.concatenate((gesture, mouse), axis=None)
    q1 = np.percentile(combined, 25)
    q3 = np.percentile(combined, 75)
    iqr = q3 - q1
    return q1 - config.iqr_multiplier * iqr, q3 + config.iqr_multiplier * iqr


def apply_iqr_filter(data: ParticipantData, config: AnalysisConfig) -> ParticipantData:
    """Published method: per-participant IQR filter (multiplier 3), with every
    out-of-range value replaced by the boundary value in BOTH conditions."""
    n_participants = config.n_participants
    n_metrics = config.n_variables_per_condition
    filtered: ParticipantData = [
        [np.array([]) for _ in range(2 * n_metrics)] for _ in range(n_participants)
    ]
    for metric_index in range(n_metrics):
        for participant_index in range(n_participants):
            gesture = np.asarray(data[participant_index][metric_index], dtype=float)
            mouse = np.asarray(data[participant_index][metric_index + n_metrics], dtype=float)
            if len(gesture) + len(mouse) == 0:
                continue
            lower, upper = _bounds(gesture, mouse, config)
            filtered[participant_index][metric_index] = np.clip(gesture, lower, upper)
            filtered[participant_index][metric_index + n_metrics] = np.clip(mouse, lower, upper)
    return filtered


def apply_original_asymmetric_filter(
    data: ParticipantData, config: AnalysisConfig
) -> ParticipantData:
    """Original notebook logic (2023), kept for documentation only: gesture values
    out of range are replaced by the boundary, mouse values out of range are deleted.
    This is the error corrected on 23-11-2026. Do not use for new analyses."""
    n_participants = config.n_participants
    n_metrics = config.n_variables_per_condition
    filtered: ParticipantData = [
        [np.array([]) for _ in range(2 * n_metrics)] for _ in range(n_participants)
    ]
    for metric_index in range(n_metrics):
        for participant_index in range(n_participants):
            gesture = np.asarray(data[participant_index][metric_index], dtype=float)
            mouse = np.asarray(data[participant_index][metric_index + n_metrics], dtype=float)
            if len(gesture) + len(mouse) == 0:
                continue
            lower, upper = _bounds(gesture, mouse, config)
            filtered[participant_index][metric_index] = np.clip(gesture, lower, upper)
            filtered[participant_index][metric_index + n_metrics] = mouse[
                (mouse >= lower) & (mouse <= upper)
            ]
    return filtered
