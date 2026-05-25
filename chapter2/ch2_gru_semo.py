"""
=============================================================================
Chapter 2 -- GRU Day-Ahead Electricity Price Forecasting (Irish SEM)
=============================================================================
Day-Ahead Electricity Price Forecasting on the Irish Single Electricity
Market (SEM) using GRU Neural Networks

Author:  Junior Kinyanzui
Project: TU Dublin TU821-4 Final Year Project
Chapter: 2 -- GRU-Based Day-Ahead Price Forecasting

=============================================================================
PURPOSE
=============================================================================
This script implements a Gated Recurrent Unit (GRU) neural network for
48-step-ahead (24-hour) day-ahead electricity price forecasting on the
Irish Single Electricity Market (I-SEM).

The model predicts all 48 half-hourly prices for the next day in a single
forward pass (multi-step direct forecasting), which matches the operational
requirement of the day-ahead market: all prices for delivery day D must be
forecast before gate closure at 11:00 on day D-1.

=============================================================================
METHODOLOGY
=============================================================================
1. HYPERPARAMETER TUNING (Section 3)
   - Keras Tuner RandomSearch (15 trials): grid over units/layers/dropout/lr
   - Optuna Bayesian Optimization (30 trials, TPE sampler): smarter search
   - Both evaluated on a held-out validation subset of training data
   - Winner selected by lowest validation MAE in EUR/MWh

2. STATIC SPLIT EVALUATION (Section 4)
   - Standard 80/20 temporal train/test split
   - Single model trained on all training data
   - Provides comparable baseline to other published work

3. WALK-FORWARD EVALUATION (Section 5) -- PRIMARY METHODOLOGY
   - Monthly sliding window: 4-year training window, 1-month test step
   - Window advances 1 month at a time through the test period
   - Warm-start: each month's model is initialised from previous month's
     weights (fine-tuning for 20 epochs vs full 50 epochs from scratch)
   - This is the most realistic evaluation: simulates production deployment
     where the model is retrained monthly on the most recent 4 years

4. ERROR ANALYSIS (Section 7)
   - Temporal decomposition: MAE by hour-of-day, month, day-of-week
   - Extreme price performance: errors during p95+ price events
   - Identifies systematic weaknesses (morning ramp, price spikes)

5. MACRO-ECONOMIC EVENT ANALYSIS (Section 8)
   - Performance during known market disruptions:
     COVID-19 lockdowns, Ukraine energy crisis, Storm Eowyn, etc.
   - Tests whether the GRU adapts to structural breaks

=============================================================================
MODEL ARCHITECTURE
=============================================================================
    Input:  (batch, 48 timesteps, N features) -- 24h of half-hourly history
    GRU:    1-2 layers, 32-128 units each, with dropout
    Output: Dense(48) -- 48 half-hourly price predictions (next 24h)

    Loss:    MSE (mean squared error) on scaled prices
    Optimizer: Adam with configurable learning rate
    Training: EarlyStopping on validation loss (patience=5)

=============================================================================
BASELINE
=============================================================================
    Naive 24-hour lag: tomorrow's price = today's price at the same time
    This is a strong baseline for electricity prices due to their strong
    daily autocorrelation. The GRU must beat this to demonstrate value.

=============================================================================
INPUTS
=============================================================================
    dataset_wind_only.csv           -- From semo_data_pipeline.py
    dataset_wind_and_total_gen.csv  -- From semo_full_power_data_pipeline.py

    Set ACTIVE_DATASET to choose which feature set to use.

=============================================================================
OUTPUTS
=============================================================================
    results/ch2/figures/    -- All visualisations (SVG + PNG at 300 DPI)
    results/ch2/tables/     -- CSV tables (monthly results, predictions, etc.)
    results/ch2/summary.json -- Machine-readable results summary
    results/ch2/summary.md  -- Human-readable results summary (for thesis)

=============================================================================
DEPENDENCIES
=============================================================================
    pip install tensorflow numpy pandas scikit-learn matplotlib optuna keras-tuner

=============================================================================
USAGE
=============================================================================
    # Run with wind + total generation features (recommended):
    python ch2_gru_semo.py

    # To run wind-only variant, edit ACTIVE_DATASET = 'wind_only' below

    Typical runtime: 2-6 hours (depending on GPU availability and tuning)

=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Suppress TensorFlow info messages and sklearn warnings for clean output
warnings.filterwarnings("ignore")

# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================
# All tuneable parameters are defined here for easy modification.
# This section acts as the single source of truth for experiment settings.

# --- Dataset Selection ---
# Two dataset variants are available, produced by the respective pipelines:
#   'wind_only': price + wind + load features (semo_data_pipeline.py)
#   'wind_and_total_gen': adds total generation features (semo_full_power_data_pipeline.py)
DATASETS = {
    'wind_only':          'dataset_wind_only.csv',
    'wind_and_total_gen': 'dataset_wind_and_total_gen.csv',
}
ACTIVE_DATASET = 'wind_and_total_gen'   # Change this to switch between experiments

# --- Forecasting Parameters ---
TARGET_COL   = "price_eur_mwh"  # Target variable to predict
LOOKBACK     = 48        # Input sequence length: 48 periods = 24 hours of history
FORECAST_H   = 48        # Output length: 48 periods = predict next 24 hours
STEP_SIZE    = FORECAST_H  # Non-overlapping daily predictions (no overlap between windows)
TRAIN_SPLIT  = 0.8       # 80% train / 20% test for static split evaluation

# --- Walk-Forward Configuration ---
WINDOW_YEARS = 4         # Training window size: 4 years of historical data
WARM_START   = True      # If True: fine-tune previous month's model (transfer learning)
                         # If False: train from scratch each month (cold start)

# --- Default GRU Hyperparameters ---
# These serve as fallback if tuning fails; overridden by tuning results
DEFAULT_CONFIG = {
    'gru_units': 64,       # Hidden units per GRU layer
    'n_layers': 1,         # Number of stacked GRU layers
    'dropout': 0.2,        # Dropout rate between layers (regularisation)
    'learning_rate': 0.001,  # Adam optimizer learning rate
    'batch_size': 64,      # Mini-batch size for training
    'epochs': 50,          # Maximum training epochs (EarlyStopping may stop earlier)
}

# Will be updated after hyperparameter tuning completes
BEST_CONFIG = DEFAULT_CONFIG.copy()

# --- Reproducibility ---
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# --- Output Directories ---
# Resolve relative to project root (one level up from Chapter 2/)
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)          # Parent of 'Chapter 2'
FIG_DIR     = os.path.join(PROJECT_DIR, 'results', 'ch2', 'figures')
TABLE_DIR   = os.path.join(PROJECT_DIR, 'results', 'ch2', 'tables')
CH2_DIR     = os.path.join(PROJECT_DIR, 'results', 'ch2')
os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# --- Macro-Economic Event Periods ---
# These are documented market disruptions affecting Irish/European electricity prices.
# Used in Section 8 to analyse model performance during structural breaks.
MACRO_EVENTS = {
    'COVID-19 Lockdowns':          ('2020-03-12', '2020-06-30'),
    'Post-COVID Recovery':         ('2020-07-01', '2021-06-30'),
    'Generation Adequacy Alerts':  ('2021-07-01', '2021-12-31'),
    'Ukraine Energy Crisis':       ('2022-02-24', '2022-12-31'),
    'EU Energy Price Caps':        ('2022-12-01', '2023-06-30'),
    'Price Normalization':         ('2023-07-01', '2024-06-30'),
    'Storm Eowyn (Ireland)':       ('2025-01-22', '2025-01-31'),
    'Winter 2024-25 Price Spike':  ('2024-12-01', '2025-02-28'),
}

# --- Figure Style Guide ---
# Consistent colour scheme across all Chapter 2 visualisations
COLORS = {
    'gru':        '#4682B4',   # Steel blue -- GRU predictions
    'actual':     '#FF8C00',   # Dark orange -- actual prices
    'naive':      '#FF7F50',   # Coral -- naive baseline
    'wind':       '#228B22',   # Forest green -- wind data
    'total_gen':  '#9370DB',   # Medium purple -- total generation
    'error':      '#DC143C',   # Crimson -- error indicators
}
EVENT_COLORS = {
    'COVID-19 Lockdowns':         '#FFB3BA',
    'Post-COVID Recovery':        '#BAFFC9',
    'Generation Adequacy Alerts': '#FFE4B5',
    'Ukraine Energy Crisis':      '#BAE1FF',
    'EU Energy Price Caps':       '#E8BAFF',
    'Price Normalization':        '#FFFFBA',
    'Storm Eowyn (Ireland)':      '#FFC0CB',
    'Winter 2024-25 Price Spike': '#D4E6F1',
}

# Matplotlib global style settings for thesis-quality figures
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': 'lightgray',
    'figure.dpi': 100,
})


def save_thesis_fig(name, fig=None):
    """
    Save a matplotlib figure in both SVG and PNG formats for the thesis.

    SVG is used for scalable vector graphics in digital submissions;
    PNG at 300 DPI meets print quality requirements.

    Parameters
    ----------
    name : str
        Base filename (without extension).
    fig : matplotlib.figure.Figure, optional
        Figure to save. Defaults to current figure (plt.gcf()).
    """
    if fig is None:
        fig = plt.gcf()
    fig.savefig(os.path.join(FIG_DIR, f'{name}.svg'), format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}.svg + {name}.png")


# --- Print experiment configuration ---
print("=" * 70)
print("  Chapter 2 -- GRU Day-Ahead Price Forecasting (Irish SEM)")
print("=" * 70)
print(f"  Active dataset: {ACTIVE_DATASET}")
print(f"  Lookback: {LOOKBACK} periods ({LOOKBACK//2}h)")
print(f"  Forecast: {FORECAST_H} periods ({FORECAST_H//2}h)")
print(f"  Warm-start: {WARM_START}")
print(f"  Seed: {SEED}")
print()


# =============================================================================
# SECTION 2: DATA LOADING & STATIC SPLIT
# =============================================================================
# Load the feature-engineered dataset and prepare it for model training.
# The static 80/20 split provides a simple baseline evaluation before
# the more rigorous walk-forward analysis.

print("\n" + "=" * 70)
print("  SECTION 2: Data Loading & Static Split")
print("=" * 70)

# Locate dataset file (check current dir and parent dir)
data_path = DATASETS[ACTIVE_DATASET]
if not os.path.exists(data_path):
    data_path = os.path.join('..', DATASETS[ACTIVE_DATASET])
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {DATASETS[ACTIVE_DATASET]}\n"
            f"Run the data pipeline first:\n"
            f"  Wind-only:     python semo_data_pipeline.py\n"
            f"  Wind+TotalGen: python semo_full_power_data_pipeline.py"
        )

# Load dataset with datetime index
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
print(f"  Loaded: {data_path}")
print(f"  Shape:      {df.shape}")
print(f"  Date range: {df.index.min().date()} -> {df.index.max().date()}")
print(f"  Columns:    {df.columns.tolist()}")

# Separate target from features
feature_cols = [c for c in df.columns if c != TARGET_COL]
all_cols     = [TARGET_COL] + feature_cols  # Target first for consistent indexing
n_features   = len(feature_cols)
print(f"  Features:   {n_features}")

# --- Static 80/20 Temporal Split ---
# IMPORTANT: This is a TEMPORAL split, not random. The test set is always
# the most recent 20% of data. This prevents future information leaking
# into training and provides a realistic evaluation scenario.
split_idx = int(len(df) * TRAIN_SPLIT)
train_df  = df.iloc[:split_idx]
test_df   = df.iloc[split_idx:]

print(f"\n  Train: {len(train_df):,} rows  ({train_df.index.min().date()} -> {train_df.index.max().date()})")
print(f"  Test:  {len(test_df):,} rows  ({test_df.index.min().date()} -> {test_df.index.max().date()})")
# Verify no temporal overlap between train and test
assert train_df.index.max() < test_df.index.min(), "Overlap detected!"
print("  No overlap between train and test")

# --- Naive Baseline Computation ---
# The naive forecast predicts tomorrow's price = today's price at the same hour.
# This is equivalent to a 24-hour (48-period) lag. It is a strong baseline
# because electricity prices exhibit strong daily autocorrelation.
test_actual    = test_df[TARGET_COL].values
naive_pred     = test_df[TARGET_COL].shift(FORECAST_H).dropna().values
actual_trimmed = test_actual[FORECAST_H:]  # Align with naive predictions

naive_mae  = mean_absolute_error(actual_trimmed, naive_pred)
naive_rmse = np.sqrt(mean_squared_error(actual_trimmed, naive_pred))

print(f"\n  Naive Baseline (24h lag):")
print(f"    MAE:  EUR {naive_mae:.2f}/MWh")
print(f"    RMSE: EUR {naive_rmse:.2f}/MWh")

# --- MinMaxScaler Fitting ---
# Scale all features to [0, 1] range for GRU training stability.
# CRITICAL: Scaler is fit on TRAINING data only to prevent information leakage.
# Test data is transformed using the training-derived scaler parameters.
scaler = MinMaxScaler()
scaler.fit(train_df[all_cols])

train_scaled = pd.DataFrame(
    scaler.transform(train_df[all_cols]),
    columns=all_cols, index=train_df.index
)
test_scaled = pd.DataFrame(
    scaler.transform(test_df[all_cols]),
    columns=all_cols, index=test_df.index
)
print(f"  Scaled {len(all_cols)} columns (fit on train only)")


# --- Sequence Builder ---
def build_sequences(data, lookback, forecast_h, target_col):
    """
    Convert a flat scaled DataFrame into 3D input/output sequences for the GRU.

    Creates sliding windows of length `lookback` as inputs (X) and the next
    `forecast_h` target values as outputs (y). This implements the
    sequence-to-sequence forecasting paradigm.

    Parameters
    ----------
    data : pd.DataFrame
        Scaled DataFrame with all features.
    lookback : int
        Number of historical timesteps to use as input (48 = 24 hours).
    forecast_h : int
        Number of future timesteps to predict (48 = 24 hours).
    target_col : str
        Name of the target column in the DataFrame.

    Returns
    -------
    X : np.ndarray, shape (n_samples, lookback, n_features)
        3D input tensor for GRU.
    y : np.ndarray, shape (n_samples, forecast_h)
        2D output matrix (scaled prices to predict).

    Example
    -------
    For lookback=48, forecast_h=48, a dataset of 1000 timesteps yields:
        X[0] = timesteps 0-47 (all features)
        y[0] = timesteps 48-95 (price only)
        X[1] = timesteps 1-48
        y[1] = timesteps 49-96
        ...
    """
    X, y = [], []
    target_idx = data.columns.get_loc(target_col)
    values     = data.values
    for i in range(lookback, len(values) - forecast_h + 1):
        X.append(values[i - lookback : i])        # Input: lookback window of all features
        y.append(values[i : i + forecast_h, target_idx])  # Output: forecast_h prices
    return np.array(X), np.array(y)


# Build training and test sequences
X_train, y_train = build_sequences(train_scaled, LOOKBACK, FORECAST_H, TARGET_COL)
X_test,  y_test  = build_sequences(test_scaled,  LOOKBACK, FORECAST_H, TARGET_COL)

print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")


def inverse_transform_prices(scaled_preds, sc, cols, target_col, n_periods):
    """
    Inverse-transform scaled price predictions back to EUR/MWh.

    Because MinMaxScaler was fit on ALL columns jointly, we must create
    a dummy array with the correct number of columns, place the scaled
    prices in the target column position, inverse-transform, then extract
    the target column.

    Parameters
    ----------
    scaled_preds : np.ndarray, shape (n_samples, n_periods)
        Scaled price predictions from the model.
    sc : MinMaxScaler
        Fitted scaler (from training data).
    cols : list of str
        Column names matching the scaler's fit order.
    target_col : str
        Name of the target column.
    n_periods : int
        Number of forecast periods (48).

    Returns
    -------
    np.ndarray, shape (n_samples, n_periods)
        Predictions in original EUR/MWh scale.
    """
    target_idx = cols.index(target_col)
    results = []
    for row in scaled_preds:
        # Create zero-filled dummy with correct number of columns
        dummy = np.zeros((n_periods, len(cols)))
        # Place scaled predictions in the target column
        dummy[:, target_idx] = row
        # Inverse transform the entire array
        inv = sc.inverse_transform(dummy)
        # Extract only the target column values
        results.append(inv[:, target_idx])
    return np.array(results)


# =============================================================================
# SECTION 3: HYPERPARAMETER TUNING
# =============================================================================
# Two methods are compared for finding optimal GRU hyperparameters:
#   1. RandomSearch (Keras Tuner): uniformly samples from a discrete grid
#   2. Bayesian Optimization (Optuna TPE): learns which regions are promising
#
# Both are evaluated on a small validation subset of training data to be fast.
# The winner (lowest validation MAE) is used for all subsequent evaluations.

print("\n" + "=" * 70)
print("  SECTION 3: Hyperparameter Tuning")
print("=" * 70)

# Use the last 20% of training sequences for tuning (most recent = most relevant)
tune_size  = int(len(X_train) * 0.20)
X_tune     = X_train[-tune_size:]
y_tune     = y_train[-tune_size:]
# Further split tuning data into train/val (90/10)
val_size   = int(len(X_tune) * 0.10)
X_tune_tr, X_tune_val = X_tune[:-val_size], X_tune[-val_size:]
y_tune_tr, y_tune_val = y_tune[:-val_size], y_tune[-val_size:]
print(f"  Tuning train: {X_tune_tr.shape[0]:,}  val: {X_tune_val.shape[0]:,}")


def build_gru_from_config(config, n_feat):
    """
    Build a GRU model from a hyperparameter configuration dictionary.

    Architecture:
        - n_layers stacked GRU layers (each with gru_units hidden units)
        - Dropout after each GRU layer (regularisation)
        - Dense output layer with FORECAST_H units (one per prediction period)

    For multi-layer GRUs, all layers except the last use return_sequences=True
    to pass the full sequence to the next GRU layer. The final GRU layer
    outputs only the last hidden state.

    Parameters
    ----------
    config : dict
        Keys: gru_units, n_layers, dropout, learning_rate, batch_size, epochs.
    n_feat : int
        Number of input features (columns in the dataset minus target).

    Returns
    -------
    tensorflow.keras.Model
        Compiled GRU model ready for training.
    """
    m = Sequential()
    for i in range(config['n_layers']):
        # Only return full sequences for intermediate layers (not the last)
        return_seq = (i < config['n_layers'] - 1)
        if i == 0:
            # First layer needs input_shape specification
            m.add(GRU(config['gru_units'],
                      input_shape=(LOOKBACK, n_feat),
                      return_sequences=return_seq))
        else:
            m.add(GRU(config['gru_units'], return_sequences=return_seq))
        # Dropout after each GRU layer to prevent overfitting
        m.add(Dropout(config['dropout']))
    # Dense output: one neuron per forecast period (direct multi-step)
    m.add(Dense(FORECAST_H))
    m.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss='mse')
    return m


# ── 3a: Keras Tuner RandomSearch ──
# RandomSearch uniformly samples hyperparameter combinations from a predefined
# grid. Simple but can miss optimal regions in high-dimensional spaces.
print("\n  --- 3a: Keras Tuner RandomSearch (15 trials) ---")

try:
    import keras_tuner as kt

    def build_gru_kt(hp):
        """Keras Tuner model builder with hyperparameter search space."""
        units    = hp.Choice("gru_units",     [32, 64, 128])
        n_layers = hp.Int("n_layers",         min_value=1, max_value=2)
        dropout  = hp.Choice("dropout",       [0.1, 0.2, 0.3])
        lr       = hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4])

        m = Sequential()
        for i in range(n_layers):
            return_seq = (i < n_layers - 1)
            if i == 0:
                m.add(GRU(units, input_shape=(LOOKBACK, X_train.shape[2]),
                          return_sequences=return_seq))
            else:
                m.add(GRU(units, return_sequences=return_seq))
            m.add(Dropout(dropout))
        m.add(Dense(FORECAST_H))
        m.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return m

    tuner = kt.RandomSearch(
        build_gru_kt,
        objective="val_loss",
        max_trials=15,            # 15 random configurations tested
        executions_per_trial=1,   # Single execution per trial (for speed)
        overwrite=True,           # Overwrite previous tuning results
        directory="kt_gru_semo",
        project_name="gru_price_ch2",
        seed=SEED,
    )

    es_tune = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    tuner.search(
        X_tune_tr, y_tune_tr,
        epochs=20,                # Quick evaluation (20 epochs max)
        batch_size=64,
        validation_data=(X_tune_val, y_tune_val),
        callbacks=[es_tune],
        verbose=0,
    )

    # Extract best hyperparameters from RandomSearch
    rs_best_hp = tuner.get_best_hyperparameters(1)[0]
    rs_best_config = {
        'gru_units':     rs_best_hp.get("gru_units"),
        'n_layers':      rs_best_hp.get("n_layers"),
        'dropout':       rs_best_hp.get("dropout"),
        'learning_rate': rs_best_hp.get("learning_rate"),
        'batch_size':    64,
        'epochs':        50,
    }

    # Get best validation loss for comparison
    rs_best_trial = tuner.oracle.get_best_trials(1)[0]
    rs_best_val_loss = rs_best_trial.score

    # Save all trial results to CSV for thesis Table
    rs_results = []
    for trial in tuner.oracle.get_best_trials(num_trials=15):
        hp_vals = trial.hyperparameters.values
        rs_results.append({
            'val_loss':      trial.score,
            'gru_units':     hp_vals.get('gru_units'),
            'n_layers':      hp_vals.get('n_layers'),
            'dropout':       hp_vals.get('dropout'),
            'learning_rate': hp_vals.get('learning_rate'),
        })
    rs_df = pd.DataFrame(rs_results).sort_values('val_loss')
    rs_df.to_csv(os.path.join(TABLE_DIR, f'random_search_results_{ACTIVE_DATASET}.csv'), index=False)

    print(f"  RandomSearch best: units={rs_best_config['gru_units']}, "
          f"layers={rs_best_config['n_layers']}, "
          f"dropout={rs_best_config['dropout']}, "
          f"lr={rs_best_config['learning_rate']}")
    print(f"  Best val_loss (scaled MSE): {rs_best_val_loss:.6f}")

    rs_completed = True

except ImportError:
    print("  keras-tuner not installed -- skipping RandomSearch")
    rs_best_config = DEFAULT_CONFIG.copy()
    rs_best_val_loss = float('inf')
    rs_completed = False


# ── 3b: Optuna Bayesian Optimization ──
# TPE (Tree-structured Parzen Estimator) models the relationship between
# hyperparameters and performance, focusing search on promising regions.
# Generally finds better configurations than random search with fewer trials.
print("\n  --- 3b: Optuna Bayesian Optimization (30 trials, 1h timeout) ---")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        """
        Optuna objective function: train a GRU and return validation MAE.

        The objective is minimised (lower MAE = better). Each trial samples
        a different hyperparameter combination guided by the TPE sampler.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object for suggesting hyperparameters.

        Returns
        -------
        float
            Validation MAE in EUR/MWh (lower is better).
        """
        # Sample hyperparameters from the search space
        gru_units = trial.suggest_categorical('gru_units', [32, 64, 128])
        n_layers  = trial.suggest_int('n_layers', 1, 2)
        dropout   = trial.suggest_float('dropout', 0.1, 0.4, step=0.1)
        lr        = trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3])
        batch     = trial.suggest_categorical('batch_size', [32, 64])

        config = {
            'gru_units': gru_units, 'n_layers': n_layers,
            'dropout': dropout, 'learning_rate': lr,
            'batch_size': batch, 'epochs': 50,
        }

        # Build and train model with this configuration
        tf.random.set_seed(SEED)
        model = build_gru_from_config(config, X_train.shape[2])

        es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        model.fit(
            X_tune_tr, y_tune_tr,
            epochs=20,
            batch_size=batch,
            validation_data=(X_tune_val, y_tune_val),
            callbacks=[es],
            verbose=0,
        )

        # Evaluate in real units (EUR/MWh) for interpretable comparison
        y_pred_s = model.predict(X_tune_val, verbose=0)
        y_pred_real = inverse_transform_prices(y_pred_s, scaler, all_cols, TARGET_COL, FORECAST_H)
        y_true_real = inverse_transform_prices(y_tune_val, scaler, all_cols, TARGET_COL, FORECAST_H)

        val_mae = mean_absolute_error(y_true_real.flatten(), y_pred_real.flatten())
        return val_mae

    # Create study with TPE sampler (Bayesian optimisation)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    # Run optimisation: 30 trials or 1-hour timeout (whichever comes first)
    study.optimize(objective, n_trials=30, timeout=3600, show_progress_bar=False)

    # Extract best configuration
    optuna_best = study.best_params
    optuna_best_config = {
        'gru_units':     optuna_best['gru_units'],
        'n_layers':      optuna_best['n_layers'],
        'dropout':       round(optuna_best['dropout'], 1),
        'learning_rate': optuna_best['learning_rate'],
        'batch_size':    optuna_best['batch_size'],
        'epochs':        50,
    }
    optuna_best_mae = study.best_value

    print(f"  Optuna best: units={optuna_best_config['gru_units']}, "
          f"layers={optuna_best_config['n_layers']}, "
          f"dropout={optuna_best_config['dropout']}, "
          f"lr={optuna_best_config['learning_rate']}, "
          f"batch={optuna_best_config['batch_size']}")
    print(f"  Best validation MAE: EUR {optuna_best_mae:.2f}/MWh")
    print(f"  Trials completed: {len(study.trials)}")

    # Save Optuna trial history
    optuna_df = study.trials_dataframe()
    optuna_df.to_csv(os.path.join(TABLE_DIR, f'optuna_study_{ACTIVE_DATASET}.csv'), index=False)

    # Save best parameters as JSON for reproducibility
    with open(os.path.join(CH2_DIR, f'optuna_best_params_{ACTIVE_DATASET}.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'best_params': optuna_best_config,
            'best_mae_eur_mwh': round(optuna_best_mae, 2),
            'n_trials': len(study.trials),
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    optuna_completed = True

except ImportError:
    print("  optuna not installed -- skipping Bayesian optimization")
    print("  Install with: pip install optuna")
    optuna_best_config = DEFAULT_CONFIG.copy()
    optuna_best_mae = float('inf')
    optuna_completed = False


# ── 3c: Method Comparison -- Select the Winner ──
print("\n  --- 3c: RandomSearch vs Optuna Comparison ---")

# Select the configuration that achieved the lowest validation MAE
if optuna_completed:
    best_method = 'optuna'
    BEST_CONFIG = optuna_best_config.copy()
    print(f"  Winner: Optuna (MAE EUR {optuna_best_mae:.2f}/MWh)")
else:
    best_method = 'random_search'
    BEST_CONFIG = rs_best_config.copy()
    print(f"  Winner: RandomSearch (only method available)")

# Save comparison table for thesis
comparison = {
    'random_search': {
        'config': rs_best_config if rs_completed else 'N/A',
        'val_loss_scaled': float(rs_best_val_loss) if rs_completed else None,
        'trials': 15 if rs_completed else 0,
    },
    'optuna': {
        'config': optuna_best_config if optuna_completed else 'N/A',
        'val_mae_eur_mwh': float(optuna_best_mae) if optuna_completed else None,
        'trials': len(study.trials) if optuna_completed else 0,
    },
    'winner': best_method,
    'best_config': BEST_CONFIG,
}

comp_rows = []
if rs_completed:
    comp_rows.append({
        'method': 'RandomSearch',
        'gru_units': rs_best_config['gru_units'],
        'n_layers': rs_best_config['n_layers'],
        'dropout': rs_best_config['dropout'],
        'learning_rate': rs_best_config['learning_rate'],
        'batch_size': rs_best_config['batch_size'],
        'trials': 15,
        'metric': 'val_loss (scaled MSE)',
        'best_score': rs_best_val_loss,
    })
if optuna_completed:
    comp_rows.append({
        'method': 'Optuna (TPE)',
        'gru_units': optuna_best_config['gru_units'],
        'n_layers': optuna_best_config['n_layers'],
        'dropout': optuna_best_config['dropout'],
        'learning_rate': optuna_best_config['learning_rate'],
        'batch_size': optuna_best_config['batch_size'],
        'trials': len(study.trials),
        'metric': 'val MAE (EUR/MWh)',
        'best_score': optuna_best_mae,
    })

comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(os.path.join(TABLE_DIR, f'tuning_method_comparison_{ACTIVE_DATASET}.csv'), index=False)
print(f"  Saved: tuning_method_comparison.csv")

print(f"\n  BEST CONFIG for all subsequent runs:")
for k, v in BEST_CONFIG.items():
    print(f"    {k}: {v}")


# =============================================================================
# SECTION 4: STATIC SPLIT EVALUATION
# =============================================================================
# Train the GRU on the full training set (80%) and evaluate on test set (20%).
# This provides a single-number performance comparison against the naive baseline
# and published benchmarks in the literature.

print("\n" + "=" * 70)
print("  SECTION 4: Static Split Evaluation")
print("=" * 70)

# Build model with best hyperparameters from tuning
tf.random.set_seed(SEED)
static_model = build_gru_from_config(BEST_CONFIG, X_train.shape[2])
static_model.summary()

# EarlyStopping: halt training when validation loss stops improving
# patience=5 means stop if no improvement for 5 consecutive epochs
es_static = EarlyStopping(
    monitor='val_loss', patience=5,
    restore_best_weights=True, verbose=1
)

# Train on full training set with 10% held out as validation
history = static_model.fit(
    X_train, y_train,
    epochs=BEST_CONFIG['epochs'],
    batch_size=BEST_CONFIG['batch_size'],
    validation_split=0.1,       # Last 10% of training data as validation
    callbacks=[es_static],
    verbose=1,
)

# --- Generate predictions on test set ---
y_pred_scaled = static_model.predict(X_test, verbose=0)
# Inverse transform: scaled [0,1] predictions -> EUR/MWh
y_pred = inverse_transform_prices(y_pred_scaled, scaler, all_cols, TARGET_COL, FORECAST_H)
y_true = inverse_transform_prices(y_test, scaler, all_cols, TARGET_COL, FORECAST_H)

# Flatten for aggregate metrics (all periods, all forecast windows)
y_pred_flat = y_pred.flatten()
y_true_flat = y_true.flatten()

# Compute performance metrics
static_mae  = mean_absolute_error(y_true_flat, y_pred_flat)
static_rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
# Percentage improvement over naive baseline
static_mae_imp  = 100 * (naive_mae - static_mae)   / naive_mae
static_rmse_imp = 100 * (naive_rmse - static_rmse) / naive_rmse

print(f"\n  Static Split Results:")
print(f"    GRU MAE:  EUR {static_mae:.2f}/MWh    RMSE: EUR {static_rmse:.2f}/MWh")
print(f"    Naive MAE: EUR {naive_mae:.2f}/MWh   RMSE: EUR {naive_rmse:.2f}/MWh")
print(f"    MAE improvement:  {static_mae_imp:.1f}%")
print(f"    RMSE improvement: {static_rmse_imp:.1f}%")

# Save static evaluation results
static_results = {
    'dataset': ACTIVE_DATASET,
    'config': BEST_CONFIG,
    'gru_mae': round(static_mae, 2),
    'gru_rmse': round(static_rmse, 2),
    'naive_mae': round(naive_mae, 2),
    'naive_rmse': round(naive_rmse, 2),
    'mae_improvement_pct': round(static_mae_imp, 1),
    'rmse_improvement_pct': round(static_rmse_imp, 1),
}
pd.DataFrame([static_results]).to_csv(
    os.path.join(TABLE_DIR, f'static_split_results_{ACTIVE_DATASET}.csv'), index=False
)

# --- Build non-overlapping datetime-indexed prediction series ---
# For plotting, we sample every FORECAST_H-th prediction to avoid overlaps
step = FORECAST_H
sampled_rows = range(0, len(y_pred), step)

pred_vals   = np.concatenate([y_pred[i] for i in sampled_rows])
actual_vals = np.concatenate([y_true[i] for i in sampled_rows])

# Map predictions back to their corresponding timestamps
dt_index = []
for i in sampled_rows:
    start_pos = LOOKBACK + i
    end_pos   = start_pos + FORECAST_H
    dt_index.extend(test_df.index[start_pos:end_pos])
dt_index = pd.DatetimeIndex(dt_index[:len(pred_vals)])

pred_series   = pd.Series(pred_vals[:len(dt_index)],   index=dt_index)
actual_series = pd.Series(actual_vals[:len(dt_index)], index=dt_index)

# Save predictions for external analysis
static_pred_df = pd.DataFrame({
    'actual': actual_series.values,
    'predicted': pred_series.values,
    'error': actual_series.values - pred_series.values,
}, index=dt_index)
static_pred_df.index.name = 'datetime'
static_pred_df.to_csv(os.path.join(TABLE_DIR, f'static_predictions_{ACTIVE_DATASET}.csv'))

# --- Figure: Training Loss History ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history['loss'],     label='Train loss', color=COLORS['gru'])
ax.plot(history.history['val_loss'], label='Val loss',   color=COLORS['actual'])
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss (scaled)')
ax.set_title('Static Split -- Training History')
ax.legend()
fig.tight_layout()
save_thesis_fig(f'static_split_training_history_{ACTIVE_DATASET}', fig)

# --- Figure: Full Test Period Predictions vs Actual ---
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(actual_series, label='Actual', linewidth=0.8, color=COLORS['actual'])
ax.plot(pred_series,   label='GRU Predicted', linewidth=0.8, color=COLORS['gru'], alpha=0.85)
ax.set_xlabel('Date')
ax.set_ylabel('Price (EUR/MWh)')
ax.set_title('Static Split -- Predicted vs Actual (Full Test Period)')
ax.legend(loc='upper right')
fig.autofmt_xdate()
fig.tight_layout()
save_thesis_fig(f'static_split_pred_vs_actual_full_{ACTIVE_DATASET}', fig)

# --- Figure: Zoomed First 2 Weeks ---
zoom_end = dt_index.min() + pd.Timedelta(weeks=2)
fig, ax = plt.subplots(figsize=(14, 6))
mask = dt_index <= zoom_end
ax.plot(actual_series[mask], label='Actual', linewidth=1.2, color=COLORS['actual'])
ax.plot(pred_series[mask],   label='GRU Predicted', linewidth=1.2, color=COLORS['gru'], alpha=0.85)
ax.set_xlabel('Date')
ax.set_ylabel('Price (EUR/MWh)')
ax.set_title('Static Split -- Predicted vs Actual (First 2 Weeks)')
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
save_thesis_fig(f'static_split_pred_vs_actual_2weeks_{ACTIVE_DATASET}', fig)

# --- Figure: Scatter Plot (Predicted vs Actual) ---
lim = (min(actual_series.min(), pred_series.min()) - 5,
       max(actual_series.max(), pred_series.max()) + 5)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(actual_series, pred_series, s=1, alpha=0.3, color=COLORS['gru'])
ax.plot(lim, lim, 'r--', linewidth=1.2, label='Perfect prediction')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Actual (EUR/MWh)')
ax.set_ylabel('Predicted (EUR/MWh)')
ax.set_title('Static Split -- Predicted vs Actual Scatter')
ax.legend()
fig.tight_layout()
save_thesis_fig(f'static_split_scatter_{ACTIVE_DATASET}', fig)

# --- Figure: Error Distribution Histogram ---
errors = actual_series.values - pred_series.values
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(errors, bins=80, edgecolor='black', alpha=0.7, color=COLORS['gru'])
ax.axvline(0, color=COLORS['error'], linestyle='--', linewidth=1.2)
ax.set_xlabel('Prediction Error (EUR/MWh)')
ax.set_ylabel('Frequency')
ax.set_title('Static Split -- Error Distribution')
fig.tight_layout()
save_thesis_fig(f'static_split_error_distribution_{ACTIVE_DATASET}', fig)

print("  Static split figures saved")


# =============================================================================
# SECTION 5: SLIDING WINDOW WALK-FORWARD EVALUATION
# =============================================================================
# This is the PRIMARY evaluation methodology for the thesis.
#
# Walk-forward validation simulates real-world deployment:
#   - For each test month, train on the preceding 4 years of data
#   - Make predictions for that month
#   - Advance the window by 1 month and repeat
#
# This captures:
#   - Model adaptation to evolving market conditions
#   - Concept drift (e.g., market rule changes, new generation capacity)
#   - Realistic training set size constraints
#
# WARM-START: Instead of training from scratch each month, we initialise
# from the previous month's model weights and fine-tune for 20 epochs.
# This is computationally cheaper and provides continuity between months.

print("\n" + "=" * 70)
print("  SECTION 5: Walk-Forward Evaluation (Warm-Start)")
print("=" * 70)

# Determine the test period (from static split boundary to end of data)
test_start = test_df.index.min()
test_end   = test_df.index.max()

# Generate list of month-start dates for the test period
test_months = pd.date_range(
    start=test_start.to_period("M").to_timestamp(),
    end=test_end.to_period("M").to_timestamp(),
    freq="MS"  # Month-Start frequency
)

print(f"  Test period:   {test_start.date()} -> {test_end.date()}")
print(f"  Months to run: {len(test_months)}")
print(f"  Window:        {WINDOW_YEARS} years")
print(f"  Warm-start:    {WARM_START}")
print(f"  Config:        units={BEST_CONFIG['gru_units']}, "
      f"layers={BEST_CONFIG['n_layers']}, dropout={BEST_CONFIG['dropout']}, "
      f"lr={BEST_CONFIG['learning_rate']}")


def scale_window(window_df, cols):
    """
    Fit a fresh MinMaxScaler on a specific training window.

    Each walk-forward iteration gets its own scaler fitted on its
    4-year training window. This prevents future information from
    leaking through the scaling parameters.

    Parameters
    ----------
    window_df : pd.DataFrame
        Training window data (4 years).
    cols : list of str
        Columns to scale.

    Returns
    -------
    tuple of (pd.DataFrame, MinMaxScaler)
        Scaled data and the fitted scaler (for later inverse transform).
    """
    sc = MinMaxScaler()
    sc.fit(window_df[cols])
    scaled = sc.transform(window_df[cols])
    scaled_df = pd.DataFrame(scaled, columns=cols, index=window_df.index)
    return scaled_df, sc


# Containers for collecting all walk-forward predictions
all_wf_preds   = []   # List of 48-element arrays (predicted prices)
all_wf_actuals = []   # List of 48-element arrays (actual prices)
all_wf_dates   = []   # Corresponding start timestamps
monthly_log    = []   # Monthly performance summary for thesis table
prev_model     = None # Previous month's model (for warm-start)

es_wf = EarlyStopping(
    monitor='val_loss', patience=5,
    restore_best_weights=True, verbose=0
)

# --- Main Walk-Forward Loop ---
for i, month_start in enumerate(test_months):
    # Define the test month boundaries
    month_end    = month_start + relativedelta(months=1) - pd.Timedelta("30min")
    # Training window: 4 years ending just before this month
    window_start = month_start - relativedelta(years=WINDOW_YEARS)

    print(f"  [{i+1:>2}/{len(test_months)}] "
          f"Train: {window_start.date()} -> {month_start.date()} | "
          f"Predict: {month_start.date()} -> {month_end.date()}", end=" ... ")

    # --- Step 1: Extract training window ---
    window_df = df[window_start : month_start - pd.Timedelta("30min")]
    if len(window_df) < LOOKBACK + FORECAST_H + 100:
        print("skipped (insufficient data)")
        continue

    # --- Step 2: Scale using this window's statistics only ---
    window_scaled, window_scaler = scale_window(window_df, all_cols)

    # --- Step 3: Build sequences from the training window ---
    X_w, y_w = build_sequences(window_scaled, LOOKBACK, FORECAST_H, TARGET_COL)
    if len(X_w) < 10:
        print("skipped (too few sequences)")
        continue

    # --- Step 4: Train model (warm-start or cold-start) ---
    tf.random.set_seed(SEED)

    if WARM_START and prev_model is not None:
        # WARM-START: Clone previous model architecture and copy weights,
        # then fine-tune on the new window. This provides continuity and
        # converges faster than training from scratch.
        month_model = tf.keras.models.clone_model(prev_model)
        month_model.set_weights(prev_model.get_weights())
        month_model.compile(
            optimizer=Adam(learning_rate=BEST_CONFIG['learning_rate']),
            loss='mse'
        )
        fit_epochs = 20   # Fewer epochs needed for fine-tuning
    else:
        # COLD-START: Build a fresh model from scratch
        month_model = build_gru_from_config(BEST_CONFIG, X_train.shape[2])
        fit_epochs = BEST_CONFIG['epochs']

    month_model.fit(
        X_w, y_w,
        epochs=fit_epochs,
        batch_size=BEST_CONFIG['batch_size'],
        validation_split=0.1,
        callbacks=[es_wf],
        verbose=0,
    )

    # Save this month's model for next month's warm-start
    prev_model = month_model

    # --- Step 5: Predict every day in this test month ---
    month_df = df[month_start : month_end]
    if len(month_df) < LOOKBACK + FORECAST_H:
        print("skipped (insufficient test month data)")
        continue

    # Include lookback buffer before month start for first predictions
    combined_df = df[month_start - pd.Timedelta(hours=LOOKBACK//2) : month_end]
    # Scale using this window's scaler (ensures consistent scaling)
    combined_scaled = pd.DataFrame(
        window_scaler.transform(combined_df[all_cols]),
        columns=all_cols, index=combined_df.index
    )

    month_preds   = []
    month_actuals = []
    month_dates   = []

    # Generate non-overlapping daily predictions within this month
    for step in range(0, len(month_df) - FORECAST_H, STEP_SIZE):
        step_time = month_df.index[step]
        step_loc  = combined_scaled.index.get_loc(step_time)

        # Need at least LOOKBACK periods of history before this point
        if step_loc < LOOKBACK:
            continue

        # Extract input sequence: LOOKBACK periods ending at step_time
        X_input = combined_scaled.values[step_loc - LOOKBACK : step_loc]
        X_input = X_input.reshape(1, LOOKBACK, len(all_cols))

        # Generate 48-step prediction
        y_pred_s = month_model.predict(X_input, verbose=0)[0]
        # Get actual values for comparison
        y_true_s = combined_scaled[TARGET_COL].values[step_loc : step_loc + FORECAST_H]

        if len(y_true_s) < FORECAST_H:
            continue

        # Inverse transform predictions and actuals back to EUR/MWh
        target_idx = all_cols.index(TARGET_COL)

        dummy_p = np.zeros((FORECAST_H, len(all_cols)))
        dummy_p[:, target_idx] = y_pred_s
        pred_inv = window_scaler.inverse_transform(dummy_p)[:, target_idx]

        dummy_t = np.zeros((FORECAST_H, len(all_cols)))
        dummy_t[:, target_idx] = y_true_s
        true_inv = window_scaler.inverse_transform(dummy_t)[:, target_idx]

        month_preds.append(pred_inv)
        month_actuals.append(true_inv)
        month_dates.append(step_time)

    # --- Step 6: Compute monthly metrics ---
    if month_preds:
        all_wf_preds.extend(month_preds)
        all_wf_actuals.extend(month_actuals)
        all_wf_dates.extend(month_dates)

        m_mae  = mean_absolute_error(np.concatenate(month_actuals), np.concatenate(month_preds))
        m_rmse = np.sqrt(mean_squared_error(np.concatenate(month_actuals), np.concatenate(month_preds)))

        # Compute naive baseline for this specific month
        month_naive_pred = month_df[TARGET_COL].shift(FORECAST_H).dropna().values
        month_naive_act  = month_df[TARGET_COL].values[FORECAST_H:]
        if len(month_naive_pred) > 0:
            m_naive_mae = mean_absolute_error(month_naive_act, month_naive_pred)
        else:
            m_naive_mae = np.nan

        monthly_log.append({
            'month': str(month_start.date()),
            'mae': round(m_mae, 2),
            'rmse': round(m_rmse, 2),
            'naive_mae': round(m_naive_mae, 2) if not np.isnan(m_naive_mae) else None,
            'improvement_pct': round(100 * (m_naive_mae - m_mae) / m_naive_mae, 1) if not np.isnan(m_naive_mae) else None,
            'n_predictions': len(month_preds),
        })

        print(f"OK  MAE: EUR {m_mae:.2f}  Naive: EUR {m_naive_mae:.2f}  "
              f"Imp: {monthly_log[-1]['improvement_pct']}%")
    else:
        print("X  No predictions")

# --- Overall Walk-Forward Aggregate Metrics ---
wf_preds_flat   = np.concatenate(all_wf_preds)
wf_actuals_flat = np.concatenate(all_wf_actuals)

wf_mae  = mean_absolute_error(wf_actuals_flat, wf_preds_flat)
wf_rmse = np.sqrt(mean_squared_error(wf_actuals_flat, wf_preds_flat))
wf_mae_imp  = 100 * (naive_mae - wf_mae)   / naive_mae
wf_rmse_imp = 100 * (naive_rmse - wf_rmse) / naive_rmse

print(f"\n  Walk-Forward Results (Warm-Start={WARM_START}):")
print(f"    GRU MAE:  EUR {wf_mae:.2f}/MWh    RMSE: EUR {wf_rmse:.2f}/MWh")
print(f"    Naive MAE: EUR {naive_mae:.2f}/MWh   RMSE: EUR {naive_rmse:.2f}/MWh")
print(f"    MAE improvement:  {wf_mae_imp:.1f}%")
print(f"    RMSE improvement: {wf_rmse_imp:.1f}%")
print(f"    Prediction days:  {len(all_wf_dates)}")

# Save walk-forward results
monthly_df = pd.DataFrame(monthly_log)
monthly_df.to_csv(os.path.join(TABLE_DIR, f'walkforward_monthly_{ACTIVE_DATASET}.csv'), index=False)

wf_results = {
    'dataset': ACTIVE_DATASET,
    'warm_start': WARM_START,
    'config': BEST_CONFIG,
    'gru_mae': round(wf_mae, 2),
    'gru_rmse': round(wf_rmse, 2),
    'naive_mae': round(naive_mae, 2),
    'naive_rmse': round(naive_rmse, 2),
    'mae_improvement_pct': round(wf_mae_imp, 1),
    'rmse_improvement_pct': round(wf_rmse_imp, 1),
    'n_months': len(monthly_log),
    'n_prediction_days': len(all_wf_dates),
}
pd.DataFrame([wf_results]).to_csv(
    os.path.join(TABLE_DIR, f'walkforward_results_{ACTIVE_DATASET}.csv'), index=False
)

# Save all walk-forward predictions with timestamps
wf_dt_index = []
for i, date in enumerate(all_wf_dates):
    step_loc = df.index.get_loc(date)
    for j in range(FORECAST_H):
        if step_loc + j < len(df):
            wf_dt_index.append(df.index[step_loc + j])
wf_dt_index = pd.DatetimeIndex(wf_dt_index[:len(wf_preds_flat)])

wf_pred_df = pd.DataFrame({
    'actual': wf_actuals_flat[:len(wf_dt_index)],
    'predicted': wf_preds_flat[:len(wf_dt_index)],
    'error': wf_actuals_flat[:len(wf_dt_index)] - wf_preds_flat[:len(wf_dt_index)],
}, index=wf_dt_index)
wf_pred_df.index.name = 'datetime'
wf_pred_df.to_csv(os.path.join(TABLE_DIR, f'walkforward_predictions_{ACTIVE_DATASET}.csv'))

# --- Walk-Forward Figures ---

# Figure: Full test period predictions vs actual
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(wf_pred_df.index, wf_pred_df['actual'],    label='Actual', linewidth=0.8, color=COLORS['actual'])
ax.plot(wf_pred_df.index, wf_pred_df['predicted'], label='GRU Predicted', linewidth=0.8, color=COLORS['gru'], alpha=0.85)
ax.set_xlabel('Date')
ax.set_ylabel('Price (EUR/MWh)')
ax.set_title('Walk-Forward -- Predicted vs Actual (Full Test Period)')
ax.legend(loc='upper right')
fig.autofmt_xdate()
fig.tight_layout()
save_thesis_fig(f'walkforward_pred_vs_actual_full_{ACTIVE_DATASET}', fig)

# Figure: First 2 weeks detail
zoom_end = wf_pred_df.index.min() + pd.Timedelta(weeks=2)
mask = wf_pred_df.index <= zoom_end
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(wf_pred_df.index[mask], wf_pred_df['actual'][mask],    label='Actual', linewidth=1.2, color=COLORS['actual'])
ax.plot(wf_pred_df.index[mask], wf_pred_df['predicted'][mask], label='GRU Predicted', linewidth=1.2, color=COLORS['gru'], alpha=0.85)
ax.set_xlabel('Date')
ax.set_ylabel('Price (EUR/MWh)')
ax.set_title('Walk-Forward -- Predicted vs Actual (First 2 Weeks)')
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
save_thesis_fig(f'walkforward_pred_vs_actual_2weeks_{ACTIVE_DATASET}', fig)

# Figure: Monthly MAE bar chart (coloured by macro event period)
fig, ax = plt.subplots(figsize=(14, 6))
months = [r['month'] for r in monthly_log]
maes   = [r['mae'] for r in monthly_log]
naive_maes = [r['naive_mae'] for r in monthly_log if r['naive_mae'] is not None]

# Colour each bar by the macro event period it falls within
bar_colors = []
for m in months:
    m_date = pd.Timestamp(m)
    color = COLORS['gru']  # Default colour
    for event, (estart, eend) in MACRO_EVENTS.items():
        if pd.Timestamp(estart) <= m_date <= pd.Timestamp(eend):
            color = EVENT_COLORS.get(event, COLORS['gru'])
            break
    bar_colors.append(color)

x = range(len(months))
ax.bar(x, maes, color=bar_colors, edgecolor='black', linewidth=0.5, label='GRU MAE')
if naive_maes:
    ax.axhline(naive_mae, color=COLORS['naive'], linestyle='--', linewidth=1.5, label=f'Naive MAE (EUR {naive_mae:.0f})')
ax.set_xticks(x)
ax.set_xticklabels([m[:7] for m in months], rotation=45, ha='right')
ax.set_ylabel('MAE (EUR/MWh)')
ax.set_title('Walk-Forward -- Monthly MAE')
ax.legend()
fig.tight_layout()
save_thesis_fig(f'walkforward_monthly_mae_{ACTIVE_DATASET}', fig)

# Figure: Monthly improvement over naive baseline
fig, ax = plt.subplots(figsize=(14, 6))
imps = [r['improvement_pct'] for r in monthly_log if r['improvement_pct'] is not None]
imp_months = [r['month'][:7] for r in monthly_log if r['improvement_pct'] is not None]
colors_imp = ['green' if v > 0 else COLORS['error'] for v in imps]
ax.bar(range(len(imps)), imps, color=colors_imp, edgecolor='black', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(range(len(imp_months)))
ax.set_xticklabels(imp_months, rotation=45, ha='right')
ax.set_ylabel('MAE Improvement over Naive (%)')
ax.set_title('Walk-Forward -- Monthly Improvement over Naive Baseline')
fig.tight_layout()
save_thesis_fig(f'walkforward_monthly_improvement_{ACTIVE_DATASET}', fig)

# Figure: Static vs Walk-forward comparison bar chart
fig, ax = plt.subplots(figsize=(8, 6))
methods = ['Static Split', 'Walk-Forward\n(Warm-Start)']
mae_vals = [static_mae, wf_mae]
rmse_vals = [static_rmse, wf_rmse]
x = np.arange(len(methods))
w = 0.3
ax.bar(x - w/2, mae_vals,  w, label='MAE',  color=COLORS['gru'], edgecolor='black')
ax.bar(x + w/2, rmse_vals, w, label='RMSE', color=COLORS['actual'], edgecolor='black')
ax.axhline(naive_mae,  color=COLORS['naive'], linestyle='--', alpha=0.7, label=f'Naive MAE (EUR {naive_mae:.0f})')
ax.axhline(naive_rmse, color=COLORS['naive'], linestyle=':',  alpha=0.7, label=f'Naive RMSE (EUR {naive_rmse:.0f})')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel('EUR/MWh')
ax.set_title('Static Split vs Walk-Forward -- Performance Comparison')
ax.legend()
fig.tight_layout()
save_thesis_fig(f'walkforward_vs_static_comparison_{ACTIVE_DATASET}', fig)

print("  Walk-forward figures saved")


# =============================================================================
# SECTION 6: FEATURE COMPARISON (PLACEHOLDER)
# =============================================================================
# To compare wind-only vs wind+total_gen models, run this script twice:
#   1. With ACTIVE_DATASET = 'wind_only'
#   2. With ACTIVE_DATASET = 'wind_and_total_gen'
# Then compare the saved CSV results.

print("\n" + "=" * 70)
print("  SECTION 6: Feature Comparison")
print("=" * 70)
print("  To compare wind-only vs wind+total_gen:")
print("  1. Run this script with ACTIVE_DATASET = 'wind_only'")
print("  2. Run again with ACTIVE_DATASET = 'wind_and_total_gen'")
print("  3. The generate_figures_from_results.py script will create")
print("     the comparison figure from both saved result CSVs.")
print(f"  Currently running: {ACTIVE_DATASET}")


# =============================================================================
# SECTION 7: ERROR ANALYSIS
# =============================================================================
# Systematic decomposition of prediction errors to identify when and why
# the GRU struggles. This informs thesis discussion of model limitations
# and suggestions for future work.

print("\n" + "=" * 70)
print("  SECTION 7: Error Analysis")
print("=" * 70)

# Use walk-forward predictions (more realistic than static split)
err_df = wf_pred_df.copy()
err_df['abs_error'] = err_df['error'].abs()
err_df['hour'] = err_df.index.hour
err_df['month'] = err_df.index.month
err_df['dow'] = err_df.index.dayofweek

# --- Top 10 Worst Prediction Days ---
# Identifies specific dates where the model performed worst,
# enabling manual investigation of causal market events
err_df['date'] = err_df.index.date
daily_mae = err_df.groupby('date')['abs_error'].mean().sort_values(ascending=False)
top10_worst = daily_mae.head(10)

print("\n  Top 10 Worst Prediction Days:")
top10_rows = []
for rank, (date, mae) in enumerate(top10_worst.items(), 1):
    day_data = err_df[err_df['date'] == date]
    avg_actual = day_data['actual'].mean()
    avg_pred   = day_data['predicted'].mean()
    print(f"    {rank}. {date}  MAE: EUR {mae:.2f}  Avg actual: EUR {avg_actual:.0f}  Avg pred: EUR {avg_pred:.0f}")
    top10_rows.append({
        'rank': rank, 'date': str(date),
        'daily_mae': round(mae, 2),
        'avg_actual': round(avg_actual, 2),
        'avg_predicted': round(avg_pred, 2),
        'likely_cause': '[FILL IN -- check market events for this date]',
    })

pd.DataFrame(top10_rows).to_csv(os.path.join(TABLE_DIR, f'top10_worst_days_{ACTIVE_DATASET}.csv'), index=False)

# --- Error by Hour of Day ---
# Reveals whether the model struggles at specific times (e.g., morning ramp)
hourly_mae = err_df.groupby('hour')['abs_error'].mean()

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(hourly_mae.index, hourly_mae.values, color=COLORS['gru'], edgecolor='black', linewidth=0.5)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('MAE (EUR/MWh)')
ax.set_title('Walk-Forward -- MAE by Hour of Day')
ax.set_xticks(range(0, 24))
fig.tight_layout()
save_thesis_fig(f'error_by_hour_{ACTIVE_DATASET}', fig)

# --- Error by Month/Season ---
# Reveals seasonal patterns in model accuracy (e.g., winter volatility)
monthly_err = err_df.groupby('month')['abs_error'].mean()

fig, ax = plt.subplots(figsize=(8, 6))
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax.bar(monthly_err.index, monthly_err.values, color=COLORS['gru'], edgecolor='black', linewidth=0.5)
ax.set_xlabel('Month')
ax.set_ylabel('MAE (EUR/MWh)')
ax.set_title('Walk-Forward -- MAE by Month')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
fig.tight_layout()
save_thesis_fig(f'error_by_season_{ACTIVE_DATASET}', fig)

# --- Error During Extreme Prices ---
# Price spikes are the hardest to predict due to their rare, non-linear nature
p95 = err_df['actual'].quantile(0.95)
extreme_mask = err_df['actual'] > p95
normal_mask  = ~extreme_mask

extreme_mae = err_df[extreme_mask]['abs_error'].mean()
normal_mae  = err_df[normal_mask]['abs_error'].mean()

print(f"\n  Error during extreme prices (>{p95:.0f} EUR/MWh, p95):")
print(f"    Extreme MAE: EUR {extreme_mae:.2f}/MWh  ({extreme_mask.sum():,} periods)")
print(f"    Normal MAE:  EUR {normal_mae:.2f}/MWh  ({normal_mask.sum():,} periods)")
print(f"    Ratio: {extreme_mae/normal_mae:.1f}x worse during extremes")

# Save error analysis results
error_analysis = {
    'hourly_mae': hourly_mae.to_dict(),
    'monthly_mae': monthly_err.to_dict(),
    'extreme_mae': round(extreme_mae, 2),
    'normal_mae': round(normal_mae, 2),
    'p95_threshold': round(p95, 2),
}
with open(os.path.join(CH2_DIR, f'error_analysis_{ACTIVE_DATASET}.json'), 'w', encoding='utf-8') as f:
    json.dump(error_analysis, f, indent=2, default=str)

print("  Error analysis figures saved")


# =============================================================================
# SECTION 8: MACRO-ECONOMIC EVENT ANALYSIS
# =============================================================================
# Analyses model performance during known market disruptions to test
# whether the GRU can adapt to structural breaks and extreme conditions.
# This is a key section for the thesis discussion chapter.

print("\n" + "=" * 70)
print("  SECTION 8: Macro-Economic Event Analysis")
print("=" * 70)

event_results = []

for event, (estart, eend) in MACRO_EVENTS.items():
    estart_dt = pd.Timestamp(estart)
    eend_dt   = pd.Timestamp(eend)

    # Filter walk-forward predictions falling within this event window
    mask = (err_df.index >= estart_dt) & (err_df.index <= eend_dt)
    event_data = err_df[mask]

    if len(event_data) == 0:
        print(f"  {event}: No walk-forward predictions in this period")
        event_results.append({
            'event': event, 'period': f'{estart} -> {eend}',
            'mae': None, 'rmse': None, 'naive_mae': None,
            'n_periods': 0, 'note': 'Outside walk-forward test period',
        })
        continue

    # Compute GRU performance during this event
    e_mae  = event_data['abs_error'].mean()
    e_rmse = np.sqrt((event_data['error'] ** 2).mean())

    # Compute naive baseline for the same period
    event_raw = df[estart_dt:eend_dt]
    if len(event_raw) > FORECAST_H:
        e_naive_pred = event_raw[TARGET_COL].shift(FORECAST_H).dropna().values
        e_naive_act  = event_raw[TARGET_COL].values[FORECAST_H:]
        e_naive_mae  = mean_absolute_error(e_naive_act, e_naive_pred)
    else:
        e_naive_mae = np.nan

    gru_beats = 'Yes' if e_mae < e_naive_mae else 'No'

    print(f"  {event:<35} MAE: EUR {e_mae:.2f}  Naive: EUR {e_naive_mae:.2f}  "
          f"GRU beats naive: {gru_beats}  ({len(event_data):,} periods)")

    event_results.append({
        'event': event,
        'period': f'{estart} -> {eend}',
        'mae': round(e_mae, 2),
        'rmse': round(e_rmse, 2),
        'naive_mae': round(e_naive_mae, 2) if not np.isnan(e_naive_mae) else None,
        'gru_beats_naive': gru_beats,
        'n_periods': len(event_data),
        'vs_overall': f"{e_mae / wf_mae:.2f}x" if wf_mae > 0 else None,
    })

event_df = pd.DataFrame(event_results)
event_df.to_csv(os.path.join(TABLE_DIR, f'macro_event_performance_{ACTIVE_DATASET}.csv'), index=False)

# --- Figure: MAE During Each Event vs Overall Average ---
events_with_data = [r for r in event_results if r['mae'] is not None]
if events_with_data:
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r['event'] for r in events_with_data]
    maes  = [r['mae'] for r in events_with_data]
    colors_ev = [EVENT_COLORS.get(n, COLORS['gru']) for n in names]

    bars = ax.barh(range(len(names)), maes, color=colors_ev, edgecolor='black', linewidth=0.5)
    ax.axvline(wf_mae, color=COLORS['error'], linestyle='--', linewidth=1.5,
               label=f'Overall MAE (EUR {wf_mae:.0f})')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('MAE (EUR/MWh)')
    ax.set_title('Walk-Forward -- MAE During Macro Events vs Overall')
    ax.legend()
    fig.tight_layout()
    save_thesis_fig(f'macro_event_performance_{ACTIVE_DATASET}', fig)

    # --- Figure: Multi-panel predicted vs actual during events ---
    events_for_panel = [r for r in events_with_data if r['n_periods'] > 48]
    if events_for_panel:
        n_panels = min(len(events_for_panel), 6)
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=False)
        if n_panels == 1:
            axes = [axes]

        for idx, r in enumerate(events_for_panel[:n_panels]):
            ax = axes[idx]
            estart_dt = pd.Timestamp(MACRO_EVENTS[r['event']][0])
            eend_dt   = pd.Timestamp(MACRO_EVENTS[r['event']][1])
            mask = (err_df.index >= estart_dt) & (err_df.index <= eend_dt)
            ed = err_df[mask]

            ax.plot(ed.index, ed['actual'],    label='Actual', color=COLORS['actual'], linewidth=0.8)
            ax.plot(ed.index, ed['predicted'], label='GRU',    color=COLORS['gru'],    linewidth=0.8, alpha=0.85)
            ax.set_title(f"{r['event']} -- MAE: EUR {r['mae']:.2f}/MWh")
            ax.set_ylabel('EUR/MWh')
            ax.legend(loc='upper right', fontsize=9)

        axes[-1].set_xlabel('Date')
        fig.tight_layout()
        save_thesis_fig(f'macro_event_pred_vs_actual_{ACTIVE_DATASET}', fig)

print("  Macro event analysis saved")


# =============================================================================
# SECTION 9: RESULTS EXPORT & MASTER SUMMARY
# =============================================================================
# Consolidates all results into JSON and Markdown formats for the thesis.

print("\n" + "=" * 70)
print("  SECTION 9: Results Export")
print("=" * 70)

# --- Optuna Figures (if tuning completed) ---
if optuna_completed:
    try:
        # Figure: Optimization convergence history
        fig, ax = plt.subplots(figsize=(10, 5))
        trial_nums = [t.number for t in study.trials]
        trial_vals = [t.value for t in study.trials]
        ax.scatter(trial_nums, trial_vals, color=COLORS['gru'], s=40, zorder=3)
        # Running best (monotonically decreasing)
        running_best = []
        best_so_far = float('inf')
        for v in trial_vals:
            best_so_far = min(best_so_far, v)
            running_best.append(best_so_far)
        ax.plot(trial_nums, running_best, color=COLORS['error'], linewidth=2, label='Best so far')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Validation MAE (EUR/MWh)')
        ax.set_title('Optuna -- Optimization History')
        ax.legend()
        fig.tight_layout()
        save_thesis_fig(f'optuna_optimization_history_{ACTIVE_DATASET}', fig)

        # Figure: Hyperparameter importance (variance-based)
        if len(study.trials) >= 10:
            param_names = ['gru_units', 'n_layers', 'dropout', 'learning_rate', 'batch_size']
            importances = {}
            for pname in param_names:
                # Group trial results by parameter value
                vals_by_param = {}
                for t in study.trials:
                    pval = t.params.get(pname)
                    if pval not in vals_by_param:
                        vals_by_param[pval] = []
                    vals_by_param[pval].append(t.value)
                # Importance = variance of group means / total variance
                # High importance means the parameter strongly affects performance
                group_means = [np.mean(v) for v in vals_by_param.values()]
                total_var = np.var(trial_vals)
                group_var = np.var(group_means)
                importances[pname] = group_var / total_var if total_var > 0 else 0

            fig, ax = plt.subplots(figsize=(8, 6))
            sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            ax.barh([x[0] for x in sorted_imp], [x[1] for x in sorted_imp],
                    color=COLORS['gru'], edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Importance (variance ratio)')
            ax.set_title('Optuna -- Hyperparameter Importance')
            fig.tight_layout()
            save_thesis_fig(f'optuna_param_importance_{ACTIVE_DATASET}', fig)

    except Exception as e:
        print(f"  Warning: Could not generate Optuna figures: {e}")

# --- Master Summary (JSON) ---
summary = {
    'generated': datetime.now().isoformat(),
    'dataset': ACTIVE_DATASET,
    'data': {
        'market': 'Irish SEM (SEMO) day-ahead market',
        'bidding_zone': '10Y1001A1001A59C',
        'resolution': '30-minute (half-hourly)',
        'date_range': f'{df.index.min().date()} -> {df.index.max().date()}',
        'total_records': len(df),
        'n_features': n_features,
        'feature_list': feature_cols,
    },
    'tuning': {
        'method': best_method,
        'best_config': BEST_CONFIG,
        'optuna_best_mae': round(optuna_best_mae, 2) if optuna_completed else None,
        'optuna_trials': len(study.trials) if optuna_completed else 0,
    },
    'static_split': static_results,
    'walk_forward': wf_results,
    'monthly_breakdown': monthly_log,
    'error_analysis': error_analysis,
    'macro_events': event_results,
    'naive_baseline': {
        'mae': round(naive_mae, 2),
        'rmse': round(naive_rmse, 2),
    },
}

with open(os.path.join(CH2_DIR, f'summary_{ACTIVE_DATASET}.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"  Saved: summary.json")

# --- Master Summary (Markdown for thesis appendix) ---
md = f"""# Chapter 2 Results Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Data
- Market: Irish SEM (SEMO) day-ahead market
- Bidding zone: 10Y1001A1001A59C
- Resolution: 30-minute (half-hourly)
- Date range: {df.index.min().date()} -> {df.index.max().date()}
- Total records: {len(df):,}
- Source: ENTSO-E Transparency Platform

## Features
- Dataset: {ACTIVE_DATASET}
- Feature count: {n_features}
- Features: {', '.join(feature_cols)}
- All features use >=24h lag (day-ahead gate closure constraint)

## Hyperparameter Tuning
- Winner: {best_method}
- Best config: units={BEST_CONFIG['gru_units']}, layers={BEST_CONFIG['n_layers']}, dropout={BEST_CONFIG['dropout']}, lr={BEST_CONFIG['learning_rate']}, batch={BEST_CONFIG['batch_size']}
- Optuna best validation MAE: EUR {optuna_best_mae:.2f}/MWh ({len(study.trials) if optuna_completed else 0} trials)

## Static Split Results
| Metric | GRU | Naive | Improvement |
|--------|-----|-------|-------------|
| MAE (EUR/MWh) | {static_mae:.2f} | {naive_mae:.2f} | {static_mae_imp:.1f}% |
| RMSE (EUR/MWh) | {static_rmse:.2f} | {naive_rmse:.2f} | {static_rmse_imp:.1f}% |

## Walk-Forward Results (4yr window, monthly retrain, warm-start={WARM_START})
| Metric | GRU | Naive | Improvement |
|--------|-----|-------|-------------|
| MAE (EUR/MWh) | {wf_mae:.2f} | {naive_mae:.2f} | {wf_mae_imp:.1f}% |
| RMSE (EUR/MWh) | {wf_rmse:.2f} | {naive_rmse:.2f} | {wf_rmse_imp:.1f}% |

## Monthly Walk-Forward Breakdown
| Month | MAE | Naive MAE | Improvement |
|-------|-----|-----------|-------------|
"""
for r in monthly_log:
    md += f"| {r['month']} | EUR {r['mae']:.2f} | EUR {r['naive_mae']:.2f} | {r['improvement_pct']}% |\n"

md += f"""
## Error Analysis
- Extreme price (>p95 = EUR {p95:.0f}) MAE: EUR {extreme_mae:.2f} ({extreme_mae/normal_mae:.1f}x worse)
- Normal price MAE: EUR {normal_mae:.2f}

## Macro Event Performance
| Event | Period | MAE | vs Overall | GRU Beats Naive? |
|-------|--------|-----|------------|------------------|
"""
for r in event_results:
    if r['mae'] is not None:
        md += f"| {r['event']} | {r['period']} | EUR {r['mae']:.2f} | {r['vs_overall']} | {r['gru_beats_naive']} |\n"
    else:
        md += f"| {r['event']} | {r['period']} | N/A | N/A | N/A |\n"

md += """
## Key Findings
- [PENDING: Complete after running both datasets]

## All Figures
See results/ch2/figures/ for all SVG + PNG outputs.
"""

with open(os.path.join(CH2_DIR, f'summary_{ACTIVE_DATASET}.md'), 'w', encoding='utf-8') as f:
    f.write(md)
print(f"  Saved: summary.md")

# --- List all generated files ---
print("\n  Generated files:")
for root, dirs, files in os.walk(CH2_DIR):
    for fname in sorted(files):
        fpath = os.path.join(root, fname)
        size_kb = os.path.getsize(fpath) // 1024
        print(f"    {fpath:<60} ({size_kb} KB)")

# =============================================================================
# COMPLETION
# =============================================================================

print("\n" + "=" * 70)
print("  Chapter 2 -- COMPLETE")
print("=" * 70)
print(f"\n  Next steps:")
print(f"  1. Run data pipelines to generate both datasets")
print(f"  2. Run this script with ACTIVE_DATASET = 'wind_only'")
print(f"  3. Run again with ACTIVE_DATASET = 'wind_and_total_gen'")
print(f"  4. Run generate_figures_from_results.py for comparison figures")
