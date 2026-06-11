"""Entry point for the gesture-vs-mouse EEG classification pipeline.

Run with::

    uv run python main.py

Loads the per-participant EEG recordings, cleans and filters them, extracts the
high-activity proportion and duration-corrected max-peak features, trains a Random
Forest over many random splits, and writes the results as CSV files into the output
directory.
"""

from __future__ import annotations

from eeg_gesture_classifier.classification import (
    ClassificationResults,
    run_classification_simulations,
)
from eeg_gesture_classifier.config import AnalysisConfig
from eeg_gesture_classifier.data_loading import load_all_participants, seed_global_rngs
from eeg_gesture_classifier.features import (
    compute_average_corrected_max_peak,
    compute_means,
    compute_proportion_above_threshold,
)
from eeg_gesture_classifier.io_results import (
    ensure_output_dir,
    write_classification_report,
    write_metric_table,
)
from eeg_gesture_classifier.outliers import apply_asymmetric_iqr_filter


def run_pipeline(config: AnalysisConfig) -> ClassificationResults:
    """Run the full analysis: load, clean, extract features, classify, write results.

    The RNG is seeded once up front; feature extraction that consumes the RNG (the
    peak simulations) runs strictly before classification so the global RNG stream,
    and therefore the results, are reproducible.
    """
    seed_global_rngs(config.random_seed)
    ensure_output_dir(config)

    raw_data = load_all_participants(config)
    filtered_data = apply_asymmetric_iqr_filter(raw_data, config)

    # Deterministic features (no RNG) first, then the RNG-consuming peak simulations.
    proportion = compute_proportion_above_threshold(filtered_data, config)
    means = compute_means(filtered_data, config)
    max_peak = compute_average_corrected_max_peak(filtered_data, config)

    write_metric_table(proportion, config, "proportion_above_threshold.csv")
    write_metric_table(means, config, "mean_metric_values.csv")
    write_metric_table(max_peak, config, "average_corrected_max_peak.csv")

    results = run_classification_simulations(max_peak, proportion, config)
    write_classification_report(results, config)

    print("\n--- Classification results ---")
    print(f"Accuracy:  {results.accuracy:.4f}")
    print(f"Precision: {results.precision:.4f}")
    print(f"Recall:    {results.recall:.4f}")
    print(f"F1-score:  {results.f1_score:.4f}")
    print(f"\nResults written to {config.output_dir}/")
    return results


if __name__ == "__main__":
    run_pipeline(AnalysisConfig())
