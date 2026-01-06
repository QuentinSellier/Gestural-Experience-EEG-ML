# Gestural-Experience-EEG-ML

This repository contains the Python implementation of the machine learning analysis presented in **Essay 3: "Understanding the Impact of Gestural Interaction Technologies on the Experience"**, as part of the doctoral dissertation by Quentin Sellier (2023).

## Project Overview

The objective of this research is to objectively differentiate between **3D gestural interaction** (using Leap Motion) and **classical mouse interaction** in a consumer context (digital catalog). The code processes Electroencephalography (EEG) data to classify the interaction modality using a Random Forest approach.

## Data Source

The analysis uses data collected via a **Bitbrain "Diadem" dry-EEG headset**. The system monitors four computed metrics:
*   Engagement
*   Memorization
*   Valence
*   Workload

## Methodology

The pipeline implemented in `eeg_analysis.py` corresponds to the four main code blocks:

1.  **Data Loading & Filtering**:
    *   Extraction of specific interaction periods (removing non-shopping phases).
    *   Cleaning of non-integer values and quality filtering.

2.  **Outlier Management**:
    *   Application of an Interquartile Range (IQR) filter (outer boundary technique, multiplier = 3) to detect and impute outliers/artifacts in the signal.

3.  **Feature Extraction & Metrics**:
    *   **Proportion of High Activity**: Calculation of the percentage of data points exceeding a threshold of 100.
    *   **Duration Correction**: Since gestural sessions were generally longer (61% of total time), a **random removal simulation** (2,000 to 5,000 iterations) is applied to equalize durations before peak analysis.
    *   **Average Max Peak**: Computation of the highest peak of brain activity, averaged over the simulations.

4.  **Classification (Machine Learning)**:
    *   **Model**: Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`).
    *   **Training**: Bootstrap sampling with 100,000 iterations.
    *   **Evaluation**: Accuracy, Precision, Recall, F1-score, and Feature Importance.

## Prerequisites

The code requires the following Python libraries:

*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `xlsxwriter`

## Citation

If you use this code or methodology, please cite the original dissertation:

> Sellier, Q. (2023). _Exploring the Impact of Gestural Interaction Technologies on the Experience: A Marketing and HCI Perspective_ (Doctoral dissertation). Université catholique de Louvain, Louvain School of Management.

## License

MIT License
