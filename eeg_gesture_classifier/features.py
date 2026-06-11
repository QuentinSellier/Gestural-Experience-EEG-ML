"""Feature extraction: high-activity proportion, mean, and corrected max peak."""

from __future__ import annotations

import numpy as np

from .config import AnalysisConfig, ParticipantData


def compute_proportion_above_threshold(
    data: ParticipantData, config: AnalysisConfig
) -> np.ndarray:
    """Return the fraction of points above the high-activity threshold.

    Result shape is ``(n_participants, 8)``; empty arrays yield a proportion of 0.
    """
    result = np.zeros((config.n_participants, 2 * config.n_variables_per_condition))
    for participant_index in range(config.n_participants):
        for variable_index in range(result.shape[1]):
            values = data[participant_index][variable_index]
            if len(values) > 0:
                above = np.sum(values > config.high_activity_threshold)
                result[participant_index][variable_index] = above / len(values)
    return result


def compute_means(data: ParticipantData, config: AnalysisConfig) -> np.ndarray:
    """Return the mean of each participant/metric array.

    Result shape is ``(n_participants, 8)``; empty arrays yield a mean of 0.
    """
    result = np.zeros((config.n_participants, 2 * config.n_variables_per_condition))
    for participant_index in range(config.n_participants):
        for variable_index in range(result.shape[1]):
            values = data[participant_index][variable_index]
            if len(values) > 0:
                result[participant_index][variable_index] = np.mean(values)
    return result


def equalize_session_durations(
    data: ParticipantData, config: AnalysisConfig
) -> ParticipantData:
    """Randomly drop points from the longer session so the two conditions match length.

    One random-removal pass per call; the caller repeats it across simulations. The
    gesture sessions were generally longer, so this removes the surplus before peak
    analysis. Arrays of length <= 1 are left untouched.
    """
    n_metrics = config.n_variables_per_condition
    corrected: ParticipantData = [
        [np.copy(data[participant_index][variable_index]) for variable_index in range(2 * n_metrics)]
        for participant_index in range(config.n_participants)
    ]

    for participant_index in range(config.n_participants):
        for metric_index in range(n_metrics):
            gesture = corrected[participant_index][metric_index]
            mouse = corrected[participant_index][metric_index + n_metrics]
            len_gesture, len_mouse = len(gesture), len(mouse)

            if len_gesture > len_mouse and len_gesture > 1:
                drop = np.random.choice(len_gesture, len_gesture - len_mouse, replace=False)
                corrected[participant_index][metric_index] = np.delete(gesture, drop)
            elif len_mouse > len_gesture and len_mouse > 1:
                drop = np.random.choice(len_mouse, len_mouse - len_gesture, replace=False)
                corrected[participant_index][metric_index + n_metrics] = np.delete(mouse, drop)

    return corrected


def compute_average_corrected_max_peak(
    data: ParticipantData, config: AnalysisConfig
) -> np.ndarray:
    """Average the per-session maximum peak over many duration-equalization simulations.

    Each simulation randomly trims the longer session to match the shorter one, takes
    the max of every metric array, and the results are averaged over all simulations.
    Result shape is ``(n_participants, 8)``.
    """
    average = np.zeros((config.n_participants, 2 * config.n_variables_per_condition))

    for simulation_index in range(config.simulations_peak):
        print(
            f"Peak simulation {simulation_index + 1}/{config.simulations_peak}",
            end="\r",
            flush=True,
        )
        corrected = equalize_session_durations(data, config)
        for participant_index in range(config.n_participants):
            for variable_index in range(average.shape[1]):
                values = corrected[participant_index][variable_index]
                if len(values) > 0:
                    average[participant_index][variable_index] += np.max(values)

    average /= config.simulations_peak
    print("\nPeak analysis completed.")
    return average
