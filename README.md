# CS5228 Project – HDB Resale Price Prediction

## Overview

This project is developed as part of **NUS CS5228: Knowledge Discovery and Data Mining (AY2025/26)**.
 The goal is to **predict the resale prices of HDB flats in Singapore** based on structured attributes such as flat type, lease commencement year, and location.
 The model is trained to minimize the **Root Mean Square Error (RMSE)** between predicted and actual resale prices on the test dataset.

The dataset provided includes:

- **`train.csv`** — labeled training data with known resale prices
- **`test.csv`** — unlabeled test data for prediction
- **Auxiliary data** — additional information (e.g., distances to schools or markets) to enhance feature richness

Our pipeline performs data preprocessing, feature augmentation, and model training using **XGBoost**, with hyperparameter tuning for optimal performance.

------

## Project Structure

```
project/
│
├── data/
│   ├── auxiliary-data/            # Extra datasets used for feature engineering
│   ├── output/                    # Final model outputs and submission files
│   ├── test/                      # Processed intermediate test/train files
│   │   ├── train_processed_2.csv
│   │   ├── test_processed_2.csv
│   │   ├── train_with_all_features.csv
│   │   └── test_with_all_features.csv
│   ├── test.csv                   # Original test dataset (without resale price)
│   └── train.csv                  # Original training dataset (with resale price)
│
├── src/
│   ├── add_features.py            # Joins processed data with auxiliary data to create new features
│   ├── config.py                  # Shared configuration values and constants
│   ├── data_processing.py         # Data cleaning, transformation, and feature encoding
│   ├── hdb.py                     # Main entry script: full pipeline from raw data to baseline model
│   ├── model_train.py             # Initial XGBoost model (demo version without tuning)
│   ├── optimize_xgboost.py        # Final optimized model training and prediction
│   └── visualization.py           # Exploratory data analysis and visualization
│
└── README.md
```

------

## Workflow Description

### 1. Data Preparation

Raw datasets (`train.csv` and `test.csv`) are cleaned and encoded in `src/data_processing.py`.
 This step includes handling missing values, categorical encoding, and normalization.

### 2. Feature Engineering

`src/add_features.py` enriches each HDB record with auxiliary information such as proximity to nearby amenities (e.g., markets, schools).
 These features are merged with the processed datasets to produce:

- `train_with_all_features.csv`
- `test_with_all_features.csv`

### 3. Model Training

Two training workflows are available:

#### a) **Full Pipeline (`hdb.py`)**

Runs the complete end-to-end pipeline:

1. Data processing
2. Feature engineering
3. Baseline XGBoost model training
4. Output generation

Usage:

```
python src/hdb.py
```

> Note: This script reproduces the entire process but uses the baseline model without hyperparameter tuning.

#### b) **Optimized Training (`optimize_xgboost.py`)**

Uses the pre-processed and feature-augmented datasets from the `data/test/` directory.
 This script fine-tunes XGBoost parameters and generates the final submission file used for evaluation.

Usage:

```
python src/optimize_xgboost.py
```

> This is the **final version** used for the official submission.

### 4. Visualization and Analysis

`src/visualization.py` provides exploratory visualizations and statistical insights into the dataset, helping guide feature design and data preprocessing decisions.
 This script is **not** part of the final pipeline but supports data understanding.

------

## Output

The final prediction file (CSV format) is saved under:

```
data/output/
```

This file contains predicted `resale_price` values for the test dataset and follows the same format as `example-submission.csv`.

------

## Course Information

**Course:** CS5228 – Knowledge Discovery and Data Mining
 **Institution:** National University of Singapore (NUS)
 **Academic Year:** 2025/2026
 **Team:** CS5228 Project Group 17 