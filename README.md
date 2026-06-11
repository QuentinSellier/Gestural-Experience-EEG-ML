# Gestural-Experience-EEG-ML

This repository contains the Python implementation of the machine learning analysis presented in **Immersive E-Commerce Interfaces: Evaluating Customer Experience in a Gesture-Controlled Digital Catalog** (Sellier et al., 2025).

## Project Overview

The objective of this research is to objectively differentiate between **3D gestural interaction** (using Leap Motion) and **classical mouse interaction** in a consumer context (digital catalog). The code processes Electroencephalography (EEG) data to classify the interaction modality using a Random Forest approach.

## Data Source

The analysis uses data collected via a **Bitbrain "Diadem" dry-EEG headset**. The system monitors four computed metrics:
*   Engagement
*   Memorization
*   Valence
*   Workload

## Methodology

The analysis lives in the `eeg_gesture_classifier` package and is run through `main.py`. The pipeline has four stages, one module each:

1.  **Data Loading & Filtering** (`data_loading.py`):
    *   Extraction of the shopping phase only (`CatalogInteraction == 1`) and good-quality rows (`quality == 1`).
    *   Cleaning of non-numeric and non-finite values.

2.  **Outlier Management** (`outliers.py`):
    *   Per-participant Interquartile Range (IQR) filter (outer boundary, multiplier = 3), with the range computed on the pooled gesture+mouse signal.
    *   Gesture and mouse are handled differently by design: gesture out-of-range values are **clamped** to the boundary, while mouse out-of-range values are **dropped**.

3.  **Feature Extraction & Metrics** (`features.py`):
    *   **Proportion of High Activity**: fraction of data points exceeding a threshold of 100.
    *   **Duration Correction**: since gestural sessions were generally longer, a random-removal simulation (`simulations_peak = 2000`) equalizes durations before peak analysis.
    *   **Average Max Peak**: highest peak of brain activity, averaged over the simulations.

4.  **Classification** (`classification.py`):
    *   **Model**: Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`, 100 trees).
    *   **Features**: max peak of Engagement and Valence, plus proportion-above-threshold of Memorization and Workload.
    *   **Training**: averaged over `simulations_ml = 1000` random train/test splits.
    *   **Evaluation**: Accuracy, Precision, Recall, F1-score, confusion matrix, and feature importances.

All configuration (paths, thresholds, simulation counts, the random seed) is centralized in `eeg_gesture_classifier/config.py`. The run is seeded (`random_seed = 42`) and reproducible.

## Project Structure

```
main.py                       # entry point: runs the full pipeline
eeg_gesture_classifier/
├── config.py                 # AnalysisConfig dataclass + shared types
├── data_loading.py           # read, filter and clean the raw .xlsx files
├── outliers.py               # asymmetric IQR filter
├── features.py               # proportion, mean, duration-corrected max peak
├── classification.py         # Random Forest simulations + results
└── io_results.py             # write CSV outputs
data/                         # raw participant .xlsx files (not distributed)
output/                       # generated CSV results
```

## Prerequisites

Dependencies are managed with [uv](https://docs.astral.sh/uv/). To set up the environment and run the analysis:

```bash
uv sync                 # create the virtual environment and install dependencies
uv run python main.py   # run the full pipeline
```

The code requires the following Python libraries:

*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `openpyxl` (reading the input `.xlsx` data files via `pandas.read_excel`)

## Outputs

Results are written to the `output/` directory as CSV files:

*   `proportion_above_threshold.csv` (proportion of points above 100, per participant and metric)
*   `mean_metric_values.csv` (mean of each metric, per participant)
*   `average_corrected_max_peak.csv` (duration-corrected average max peak, per participant and metric)
*   `classification_report.csv` (accuracy, precision, recall, F1, confusion matrix, feature importances)

## Citation

If you use this code or methodology, please cite the original article:

> Sellier, Q., Poncin, I., Vande Kerkhove, C., Vanderdonckt, J. (2025). _Immersive E-Commerce Interfaces: Evaluating Customer Experience in a Gesture-Controlled Digital Catalog_ [Manuscript submitted for publication]. Preprint available at SSRN: https://ssrn.com/abstract=5405608

## License

MIT License
