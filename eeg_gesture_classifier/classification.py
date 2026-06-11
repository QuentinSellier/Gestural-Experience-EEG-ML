"""Random Forest classification of gesture vs mouse interaction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import METRIC_NAMES, AnalysisConfig


@dataclass(frozen=True)
class ClassificationResults:
    """Metrics and feature importances, averaged over the simulation runs."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_negative: float
    false_positive: float
    false_negative: float
    true_positive: float
    feature_names: list[str]
    feature_importances: np.ndarray


def feature_names(config: AnalysisConfig) -> list[str]:
    """Return descriptive names for the four classifier features, in column order."""
    peak = [f"max_peak_{METRIC_NAMES[index]}" for index in config.peak_feature_cols]
    proportion = [
        f"prop_above_threshold_{METRIC_NAMES[index]}" for index in config.prop_feature_cols
    ]
    return peak + proportion


def build_feature_matrix(
    max_peak: np.ndarray, proportion: np.ndarray, config: AnalysisConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Build the feature matrix ``X`` and labels ``y`` (gesture=1, mouse=0).

    Each participant contributes two rows: a gesture row from columns 0-3 and a mouse
    row from columns 4-7 (the ``+ n_metrics`` offset). Features are the selected max
    peak and proportion columns.
    """
    n_metrics = config.n_variables_per_condition
    rows: list[list[float]] = []
    labels: list[int] = []

    for participant_index in range(config.n_participants):
        gesture_row = [max_peak[participant_index][col] for col in config.peak_feature_cols] + [
            proportion[participant_index][col] for col in config.prop_feature_cols
        ]
        mouse_row = [
            max_peak[participant_index][col + n_metrics] for col in config.peak_feature_cols
        ] + [proportion[participant_index][col + n_metrics] for col in config.prop_feature_cols]

        rows.append(gesture_row)
        labels.append(1)
        rows.append(mouse_row)
        labels.append(0)

    return np.array(rows), np.array(labels)


def _split(
    features: np.ndarray, labels: np.ndarray, groups: np.ndarray, config: AnalysisConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train/test according to ``config.split_strategy``.

    - ``"random"``: plain row-wise split (the original paper / notebook behaviour).
    - ``"grouped"``: participant-level split via :class:`GroupShuffleSplit`, keeping
      both of a participant's rows on the same side to avoid cross-condition leakage.

    Both strategies draw from the seeded global RNG (no explicit ``random_state``).
    """
    if config.split_strategy == "grouped":
        splitter = GroupShuffleSplit(n_splits=1, test_size=config.test_size)
        train_index, test_index = next(splitter.split(features, labels, groups))
        return (
            features[train_index],
            features[test_index],
            labels[train_index],
            labels[test_index],
        )
    if config.split_strategy == "random":
        return train_test_split(features, labels, test_size=config.test_size)
    raise ValueError(f"Unknown split_strategy: {config.split_strategy!r}")


def run_classification_simulations(
    max_peak: np.ndarray, proportion: np.ndarray, config: AnalysisConfig
) -> ClassificationResults:
    """Train and evaluate a Random Forest over many random train/test splits.

    The feature matrix is deterministic, so it is built once; only the split and the
    forest consume the (seeded) global RNG. Metrics and feature importances are
    averaged over ``config.simulations_ml`` runs.
    """
    features, labels = build_feature_matrix(max_peak, proportion, config)
    names = feature_names(config)
    # Two consecutive rows per participant: groups = [0, 0, 1, 1, ..., 38, 38].
    groups = np.repeat(np.arange(config.n_participants), 2)

    totals = {key: 0.0 for key in ("accuracy", "precision", "recall", "f1")}
    confusion_totals = {key: 0 for key in ("tn", "fp", "fn", "tp")}
    importance_sum = np.zeros(len(names))

    for simulation_index in range(config.simulations_ml):
        print(
            f"ML simulation {simulation_index + 1}/{config.simulations_ml}",
            end="\r",
            flush=True,
        )
        features_train, features_test, labels_train, labels_test = _split(
            features, labels, groups, config
        )

        classifier = RandomForestClassifier(n_estimators=config.rf_n_estimators)
        classifier.fit(features_train, labels_train)
        predictions = classifier.predict(features_test)

        totals["accuracy"] += accuracy_score(labels_test, predictions)
        totals["precision"] += precision_score(labels_test, predictions, zero_division=0)
        totals["recall"] += recall_score(labels_test, predictions, zero_division=0)
        totals["f1"] += f1_score(labels_test, predictions, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(labels_test, predictions, labels=[0, 1]).ravel()
        confusion_totals["tn"] += tn
        confusion_totals["fp"] += fp
        confusion_totals["fn"] += fn
        confusion_totals["tp"] += tp

        importance_sum += classifier.feature_importances_

    n_simulations = config.simulations_ml
    confusion_grand_total = sum(confusion_totals.values())
    print("\nClassification completed.")

    return ClassificationResults(
        accuracy=totals["accuracy"] / n_simulations,
        precision=totals["precision"] / n_simulations,
        recall=totals["recall"] / n_simulations,
        f1_score=totals["f1"] / n_simulations,
        true_negative=confusion_totals["tn"] / confusion_grand_total,
        false_positive=confusion_totals["fp"] / confusion_grand_total,
        false_negative=confusion_totals["fn"] / confusion_grand_total,
        true_positive=confusion_totals["tp"] / confusion_grand_total,
        feature_names=names,
        feature_importances=importance_sum / n_simulations,
    )
