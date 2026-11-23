# Gestural-Experience-EEG-ML

This repository contains the Python code for the EEG analyses of **Immersive E-Commerce Interfaces: Evaluating Customer Experience in a Gesture-Controlled Digital Catalog** (Sellier, Poncin, Vande Kerkhove and Vanderdonckt).

## Project Overview

The study compares mid-air gestural control (Leap Motion) and classical mouse control of the same two-dimensional digital catalog, in a within-subject experiment with 39 participants. The code processes the EEG indicators recorded during both sessions, compares the two conditions with paired statistical tests, and runs a complementary Random Forest classification of the interaction modality.

## Data Source

The data were collected with a **Bitbrain "Diadem" dry-EEG headset**. The Bitbrain software provides four computed indicators, named as follows in the code and in the article:

| Code name    | Article name   |
|--------------|----------------|
| Engagement   | Engagement     |
| Memorization | Memorization   |
| Valence      | Attraction     |
| Workload     | Cognitive load |

The raw participant files and the questionnaire data are not distributed with this repository.

## Methodology

The processing pipeline lives in the `eeg_gesture_classifier` package:

1.  **Data Loading & Filtering** (`data_loading.py`):
    *   Extraction of the shopping phase only (`CatalogInteraction == 1`) and good-quality rows (`quality == 1`).
    *   Cleaning of non-numeric and non-finite values.

2.  **Outlier Management** (`outliers.py`):
    *   Per-participant Interquartile Range (IQR) boundaries (outer boundary, multiplier = 3), computed on the pooled gesture and mouse signal of each participant.
    *   Out-of-range values are replaced by the boundary value (capping), in both the gesture and the mouse session.

3.  **Feature Extraction** (`features.py`):
    *   **Proportion of high activity**: fraction of data points above a threshold of 100.
    *   **Duration correction**: gestural sessions were generally longer, so a random-removal simulation (`simulations_peak = 5000`) equalizes the durations of the two sessions before the peak is computed.
    *   **Average max peak**: highest value of each indicator, averaged over the simulations.
    *   Session mean and median of each indicator.

4.  **Classification** (`classification.py`):
    *   **Model**: Random Forest classifier (`sklearn.ensemble.RandomForestClassifier`, 100 trees, default hyperparameters).
    *   **Features**: max peak and proportion of high activity of the four indicators (8 features).
    *   **Evaluation**: participant-grouped train/test splits (20% of the participants in the test set; both sessions of a participant always stay on the same side), repeated over many random splits.

All configuration (paths, thresholds, simulation counts, the random seed) is centralized in `eeg_gesture_classifier/config.py`. Runs are seeded (`random_seed = 42`) and reproducible.

## Scripts

| Script | What it produces | Article |
|--------|------------------|---------|
| `recalcul_tableaux_eeg.py` | Per-participant summaries (peaks, proportions, means, medians); paired Wilcoxon comparisons with effect sizes for the full sample and each presentation order; Welch comparison of the gesture-minus-mouse differences between orders | Tables 4 and 5, Table B.8 (EEG rows), Table C.9 |
| `classification_check.py` | Accuracy of the 8-feature Random Forest over 10,000 participant-grouped splits, with its variability across splits and the confusion matrix (`main`); permutation test that preserves the participant pairing (`perm`) | Section 5.2, Appendix A.4 |
| `sensitivity_analysis.py` | Paired comparisons with alternative outlier boundaries (k = 1.5, 3, 5, 10) and without outlier treatment | Table C.10 |
| `main.py` | Full pipeline in one run: summaries written as CSV files and a Random Forest run with the settings of `config.py` | |

In the permutation test, the gesture and mouse labels are swapped at random within each participant, so that each participant keeps one session of each label, and the accuracy is re-estimated over participant-grouped splits. The p-value is (1 + number of permutations with an accuracy at least as high as the observed one) / (1 + number of permutations).

## Usage

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                                                        # create the environment
uv run python recalcul_tableaux_eeg.py --raw data --order <questionnaire.xlsx> --out output
uv run python classification_check.py main                     # reads output/eeg_comparisons.xlsx
uv run python classification_check.py perm --perms 200 --splits-per-perm 100
uv run python sensitivity_analysis.py --raw data --out output
uv run python main.py                                          # full pipeline with CSV outputs
```

The questionnaire file is only used to retrieve the presentation order of each participant.

The code requires the following Python libraries:

*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `scipy`
*   `openpyxl` (reading the input `.xlsx` data files via `pandas.read_excel`)

## Project Structure

```
main.py                       # full pipeline: features, CSV outputs, Random Forest run
recalcul_tableaux_eeg.py      # paired statistical comparisons (Tables 4, 5, B.8, C.9)
classification_check.py       # classification accuracy and permutation test (Appendix A.4)
sensitivity_analysis.py       # sensitivity to the outlier boundary (Table C.10)
eeg_gesture_classifier/
├── config.py                 # AnalysisConfig dataclass + shared types
├── data_loading.py           # read, filter and clean the raw .xlsx files
├── outliers.py               # IQR boundaries and capping
├── features.py               # proportion, mean, duration-corrected max peak
├── classification.py         # Random Forest simulations + results
└── io_results.py             # write CSV outputs
data/                         # raw participant .xlsx files (not distributed)
output/                       # generated results
```

## Outputs

Results are written to the `output/` directory:

*   `eeg_comparisons.xlsx` and `tables_4_5.tex` (summaries per participant and statistical comparisons)
*   `classification_check.json` and `permutation_test.json`
*   `sensitivity_outlier_boundary.csv`
*   `proportion_above_threshold.csv`, `mean_metric_values.csv`, `average_corrected_max_peak.csv` and `classification_report.csv` (from `main.py`)

## Citation

If you use this code or methodology, please cite the article:

> Sellier, Q., Poncin, I., Vande Kerkhove, C., Vanderdonckt, J. (2025). _Immersive E-Commerce Interfaces: Evaluating Customer Experience in a Gesture-Controlled Digital Catalog_ [Manuscript submitted for publication]. Preprint available at SSRN: https://ssrn.com/abstract=5405608

## License

MIT License
