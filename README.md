# GRU-Based Forecasting: Equity Drawdown Prediction and Day-Ahead Electricity Price Forecasting

**TU821 Final Year Project** | Junior John Kinyanzui (D21127732) | Supervisor: Nouman Ashraf | May 2026

This repository contains the source code for a final year project investigating whether a single Gated Recurrent Unit (GRU) architecture can extract predictive signal from sequential market data across two structurally different domains.

## Project structure

```
chapter1/                   NASDAQ-100 drawdown classification
  01_data_pipeline.ipynb      Raw data acquisition and cleaning
  02_event_definition.ipynb   Binary drawdown label construction
  03_feature_operators.ipynb  Feature engineering (VIX, realised vol, lagged returns)
  04_baselines.ipynb          Constant, logistic regression, random forest baselines
  05_GRU_architecture_bulletproof.py   GRU classifier with walk-forward evaluation
  diagnostic_gru_bulletproof.py        Hyperparameter search and MCPT significance testing

chapter2/                   Irish SEM day-ahead price forecasting
  semo_data_pipeline.py       SEMO price and EirGrid wind generation data acquisition
  semo_full_power_data_pipeline.py  Extended pipeline with total generation
  ch2_gru_semo.py             GRU regression model with walk-forward retraining

results/                    Evaluation outputs, figures, and tables
  ch1/figures/                Chapter 1 plots (fold timelines, calibration, MCPT, etc.)
  ch1/tables/                 Chapter 1 CSV results (Brier scores, baselines, holdout)
  ch2/figures/                Chapter 2 plots (predictions vs actual, error analysis, etc.)
  ch2/tables/                 Chapter 2 CSV results (walk-forward monthly, worst days, etc.)
```

## Chapter 1: Probabilistic drawdown estimation (NASDAQ-100)

A minimal-feature GRU binary classifier predicting 3-day, 3% drawdown events on the NASDAQ-100 index (2000-2026). Evaluated under strict walk-forward cross-validation with 18 tuning folds and 5 held-out folds. Statistical significance assessed via Monte Carlo Permutation Test.

**Key result:** Mean Brier score of 0.1010 vs 0.1037 (constant baseline). MCPT p-value = 0.72 --- improvement is not statistically significant.

## Chapter 2: Day-ahead electricity price forecasting (Irish SEM)

A GRU regression model producing 48 simultaneous half-hourly price predictions for the Irish Single Electricity Market. Monthly walk-forward retraining with a 4-year sliding window over 508 prediction days.

**Key result:** MAE of 24.34 EUR/MWh, a 25.1% improvement over the naive 24-hour lag baseline.

## Data sources

Raw data is not included due to redistribution licensing constraints. The data acquisition scripts will reproduce the full dataset when executed.

- **Chapter 1:** NASDAQ-100 daily closes and VIX via `yfinance`
- **Chapter 2:** Irish SEM day-ahead prices from [SEMO](https://www.sem-o.com), wind and total generation from [EirGrid Smart Grid Dashboard](https://www.smartgriddashboard.com)

## Requirements

Python 3.10+ with TensorFlow. Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproducibility

Random seeds are fixed for all stochastic operations. Each script can be run sequentially by chapter number. Notebooks (`.ipynb`) contain additional exploratory visualisations.

## License

This project is submitted in partial fulfilment of the Honours Degree in Electrical and Electronic Engineering (TU821) at Technological University Dublin.
