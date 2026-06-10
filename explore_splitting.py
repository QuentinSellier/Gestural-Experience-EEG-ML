"""Compare random vs participant-grouped train/test splitting.

The paper claims a participant-level split that prevents cross-condition leakage, but
the code performs a plain row-wise random split. This script quantifies how much the
reported accuracy depends on that choice, for the 8-, 4-, and 2-feature models.

The expensive feature computation (load, outlier filter, 2000 peak simulations) is run
once; only the cheap classification loop is repeated per configuration. The RNG is
re-seeded before each configuration so every result is independently reproducible.

Run with::

    uv run python explore_splitting.py
"""

from __future__ import annotations

from dataclasses import replace

from eeg_gesture_classifier.classification import run_classification_simulations
from eeg_gesture_classifier.config import AnalysisConfig
from eeg_gesture_classifier.data_loading import load_all_participants, seed_global_rngs
from eeg_gesture_classifier.features import (
    compute_average_corrected_max_peak,
    compute_proportion_above_threshold,
)
from eeg_gesture_classifier.outliers import apply_asymmetric_iqr_filter

# Feature sets, expressed as (max-peak columns, proportion columns) into a 4-metric
# block where 0=engagement, 1=memorization, 2=valence/attraction, 3=workload.
FEATURE_SETS = {
    "8-feature (all)": ((0, 1, 2, 3), (0, 1, 2, 3)),
    "4-feature (paper best)": ((0, 2), (1, 3)),  # peak eng+val, prop mem+work
    "2-feature (paper best)": ((2,), (3,)),       # peak attraction, prop workload
}
SPLIT_STRATEGIES = ("random", "grouped")


def main() -> None:
    base = AnalysisConfig()

    # --- Expensive stage, run once -------------------------------------------------
    seed_global_rngs(base.random_seed)
    raw = load_all_participants(base)
    filtered = apply_asymmetric_iqr_filter(raw, base)
    proportion = compute_proportion_above_threshold(filtered, base)
    max_peak = compute_average_corrected_max_peak(filtered, base)

    # --- Cheap stage: evaluate every (feature set, split strategy) ------------------
    results: dict[tuple[str, str], float] = {}
    for set_name, (peak_cols, prop_cols) in FEATURE_SETS.items():
        for strategy in SPLIT_STRATEGIES:
            config = replace(
                base,
                peak_feature_cols=peak_cols,
                prop_feature_cols=prop_cols,
                split_strategy=strategy,
            )
            print(f"\n>>> {set_name} | split={strategy}")
            seed_global_rngs(config.random_seed)  # independent, reproducible per run
            outcome = run_classification_simulations(max_peak, proportion, config)
            results[(set_name, strategy)] = outcome.accuracy

    # --- Report --------------------------------------------------------------------
    print("\n\n================ Accuracy: random vs participant-grouped ================")
    header = f"{'Model':<26}{'Random':>10}{'Grouped':>10}{'Drop':>10}"
    print(header)
    print("-" * len(header))
    for set_name in FEATURE_SETS:
        random_acc = results[(set_name, "random")]
        grouped_acc = results[(set_name, "grouped")]
        drop = random_acc - grouped_acc
        print(
            f"{set_name:<26}{random_acc:>10.4f}{grouped_acc:>10.4f}"
            f"{drop:>+10.4f}"
        )


if __name__ == "__main__":
    main()
