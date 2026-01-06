# EEG-Gesture-Interaction-ML

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

The pipeline implemented in `eeg_analysis.py` follows these steps:

1.  **Data Filtering**: 
    *   Extraction of specific interaction periods (removing non-shopping phases).
    *   Cleaning of non-integer values and quality filtering.
2.  **Outlier Management**: 
    *   Application of an Interquartile Range (IQR) filter (outer boundary technique) to detect and impute outliers/artifacts.
3.  **Duration Correction**: 
    *   Since gestural sessions were generally longer (61% of total time) than mouse sessions, a **random removal simulation** is applied.
    *   Data points are randomly removed from the longer sessions to equalize duration.
    *   This process is simulated 2,000 to 5,000 times to ensure statistical stability.
4.  **Feature Extraction**:
    *   **Average Max Peak**: The highest peak of brain activity (averaged over simulations).
    *   **Proportion of High Activity**: The percentage of data points exceeding a threshold of 100 (high cognitive/emotional intensity).
5.  **Classification**:
    *   **Model**: Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`).
    *   **Training**: Bootstrap sampling with 100,000 iterations.
    *   **Evaluation**: Accuracy, Precision, Recall, and F1-score.

## Prerequisites

The code requires the following Python libraries:

*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `xlsxwriter`

## Citation
If you use this code or methodology, please cite the original dissertation:

Sellier, Q. (2023). Exploring the Impact of Gestural Interaction Technologies on the Experience: A Marketing and HCI Perspective (Doctoral dissertation). Université catholique de Louvain, Louvain School of Management.
