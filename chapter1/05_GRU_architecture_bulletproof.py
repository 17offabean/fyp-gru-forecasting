#!/usr/bin/env python3
"""
05_GRU_architecture_bulletproof.py
==================================
FULL 12-Feature Regime-Specific GRU for NASDAQ-100 Drawdown Prediction.

Thesis: "Probabilistic Risk Estimation in Financial Markets using GRU Neural Networks"
Chapter 1: NASDAQ-100 Drawdown Prediction

Purpose:
    This script trains a regime-specific GRU neural network using ALL 12 available
    features to predict the probability of a >= 3% drawdown in the NASDAQ-100 index
    within the next 3 trading days. Three separate GRU models are trained, one for
    each VIX regime (low < 15, medium 15-25, high >= 25), under the hypothesis that
    market dynamics differ across volatility regimes.

    The results from this "kitchen-sink" model serve as a NEGATIVE comparison point:
    the minimal 4-feature GRU in diagnostic_gru_bulletproof.py is expected to
    outperform this, demonstrating the principle of feature parsimony.

Methodology:
    1. Walk-Forward Validation: Expanding-window training with 756-day train,
       252-day validation, stepping 252 days. Produces 23 total folds; first 18
       are used for tuning, last 5 held out.
    2. Regime Segmentation: At prediction time, the current VIX level determines
       which of 3 GRU models generates the probability estimate.
    3. Hyperparameter Search: 30 random configurations tested across all 18
       tuning folds (30 configs x 18 folds x 3 regimes = 1620 model fits).
    4. Evaluation Metric: Brier Score (mean squared error of probability forecasts),
       compared against a constant baseline (predicting the marginal event rate)
       and logistic regression.

Inputs:
    - data/features_^NDX.parquet: Pre-computed feature matrix with columns for
      log returns, lagged returns, VIX levels, VIX changes, and realized volatility.
      Must contain a 'label' column (binary: 1 if >= 3% drawdown within 3 days).

Outputs:
    - results/ch1/figures/full_gru_brier_by_fold.{svg,png}
    - results/ch1/figures/full_gru_best_config_brier_by_fold.{svg,png}
    - results/ch1/tables/full_gru_results.csv
    - results/ch1/tables/full_gru_hyperparam_search.csv
    - results/full_gru_hyperparameter_search_results.pkl
    - results/ch1/full_gru_summary.json

Runtime:
    Approximately 3-6 hours depending on hardware. Designed for overnight execution.

"""

# ============================================================================
# SECTION 0: ENVIRONMENT SETUP AND CONFIGURATION
# ============================================================================
# Standard library imports for memory management, serialisation, and timing
import gc
import json
import os
import random
import time

# Scientific computing stack
import numpy as np
import pandas as pd
from pathlib import Path

# Scikit-learn utilities for feature scaling and class imbalance handling
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow/Keras for GRU model construction and training
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense, Bidirectional, Input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

# Matplotlib for figure generation (non-interactive backend for headless runs)
import matplotlib
matplotlib.use('Agg')  # Must be set BEFORE importing pyplot; avoids Tkinter errors
import matplotlib.pyplot as plt

# ============================================================================
# REPRODUCIBILITY: Fix all random seeds to ensure deterministic results
# across NumPy, TensorFlow, and Python's built-in random module.
# Note: Perfect reproducibility on GPU is not guaranteed due to floating-point
# non-associativity in parallel reductions, but this minimises variation.
# ============================================================================
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- Thesis colour scheme (consistent across all Chapter 1 figures) ---
CLR_GRU      = '#4682B4'   # steelblue  - GRU model results
CLR_CONSTANT = '#FF7F50'   # coral      - constant (naive) baseline
CLR_LOGISTIC = '#FFD700'   # gold       - logistic regression baseline
CLR_NULL     = '#B0C4DE'   # lightsteelblue - null distribution (MCPT)

# --- Global matplotlib styling for publication-quality figures ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.figsize': (10, 6),
    'grid.alpha': 0.3,
    'grid.color': 'lightgray',
})

# --- Output directories (created if they don't exist) ---
FIG_DIR = os.path.join('results', 'ch1', 'figures')
TBL_DIR = os.path.join('results', 'ch1', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)
os.makedirs('results/ch1', exist_ok=True)


def save_thesis_fig(name, fig=None):
    """
    Save a matplotlib figure in dual format for thesis inclusion.

    Saves both SVG (vector, for LaTeX/Word embedding) and high-DPI PNG
    (raster, for compatibility) to the standard figures directory.

    Parameters
    ----------
    name : str
        Base filename without extension (e.g., 'full_gru_brier_by_fold').
    fig : matplotlib.figure.Figure, optional
        Figure to save. If None, uses the current active figure (plt.gcf()).
    """
    if fig is None:
        fig = plt.gcf()
    fig.savefig(os.path.join(FIG_DIR, f'{name}.svg'), format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')
    print(f"  [SAVED] {name}.svg + {name}.png")


# ============================================================================
# SECTION 1: MODEL CONFIGURATION AND DATA LOADING
# ============================================================================

# --- Baseline GRU configuration (used before hyperparameter search) ---
# These values are sensible defaults from the literature on RNNs for
# financial time series (see Che et al., 2018; Fischer & Krauss, 2018).
CFG = {
    "primary_symbol": "^NDX",       # NASDAQ-100 index ticker
    "lookback": 20,                 # Sequence length: 20 trading days (~1 month)
    "hidden_units": 32,             # GRU hidden state dimensionality
    "dropout": 0.2,                 # Dropout rate for regularisation
    "learning_rate": 0.001,         # RMSprop learning rate
    "batch_size": 32,               # Mini-batch size for SGD
    "max_epochs": 100,              # Maximum training epochs (early stopping usually triggers earlier)
    "early_stop_patience": 15,      # Epochs without val_loss improvement before stopping
}

# --- File paths ---
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# --- FULL 12-feature set ---
# This is the "kitchen-sink" feature set. All features are temporal sequences
# fed to the GRU. The hypothesis is that this is over-parameterised and the
# minimal set (VIX, VIX_5d_change, realized_vol_5d, ret_lag_1) performs better.
sequence_features = [
    'log_ret',          # Daily log return of NASDAQ-100
    'ret_lag_1',        # 1-day lagged return
    'ret_lag_2',        # 2-day lagged return
    'ret_lag_3',        # 3-day lagged return
    'ret_lag_4',        # 4-day lagged return
    'ret_lag_5',        # 5-day lagged return
    'VIX',              # CBOE Volatility Index (current level)
    'VIX_5d_change',    # 5-day percentage change in VIX
    'VIX_change_lag_1', # 1-day lagged VIX change
    'VIX_change_lag_2', # 2-day lagged VIX change
    'VIX_change_lag_3', # 3-day lagged VIX change
    'realized_vol_5d'   # 5-day realised volatility (std of returns)
]

# --- Load pre-computed feature matrix ---
features_file = DATA_DIR / f"features_{CFG['primary_symbol']}.parquet"
df = pd.read_parquet(features_file)
print(f"Loaded {len(df)} rows from {features_file}")
print(f"Feature set ({len(sequence_features)} features): {sequence_features}")


# ============================================================================
# SECTION 1B: WALK-FORWARD SPLIT GENERATION
# ============================================================================

def walkforward_splits(n, train_days=756, val_days=252, step_days=252):
    """
    Generate walk-forward (expanding window) train/validation split indices.

    This implements a time-series-appropriate cross-validation scheme that
    respects temporal ordering (no future data leakage). Each fold uses
    a fixed-size training window followed by a fixed-size validation window,
    advancing by step_days between folds.

    Parameters
    ----------
    n : int
        Total number of observations (trading days) in the dataset.
    train_days : int, default=756
        Training window size in trading days (~3 years).
    val_days : int, default=252
        Validation window size in trading days (~1 year).
    step_days : int, default=252
        Step size between consecutive folds (~1 year).

    Returns
    -------
    list of dict
        Each dict contains 'train_start', 'train_end', 'val_start', 'val_end'
        as integer indices into the DataFrame.

    Notes
    -----
    With ~6000 observations and default parameters, this generates ~23 folds.
    The first 18 (indices 0-17) are used for tuning; the remaining 5 (18-22)
    are held out for final evaluation.
    """
    splits = []
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_days
        val_start = train_end
        val_end = val_start + val_days
        # Stop when validation window would extend beyond available data
        if val_end > n:
            break
        splits.append({
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end
        })
        start += step_days
    return splits


splits = walkforward_splits(len(df))
print(f"Total walk-forward folds: {len(splits)}")

# TUNING FOLDS ONLY (0-17): Used for hyperparameter search.
# Held-out folds (18+) are reserved for final unbiased evaluation.
TUNING_FOLDS = range(0, 18)
print(f"Tuning folds: {list(TUNING_FOLDS)}")


# ============================================================================
# SECTION 2: SEQUENCE CREATION
# ============================================================================

def create_sequences(df, feature_cols, lookback=20):
    """
    Convert a DataFrame of daily features into 3D input tensors for the GRU.

    For each day t (starting from index `lookback`), creates a sequence of
    shape (lookback, n_features) from days [t-lookback, t). The label is
    taken from day t (whether a drawdown occurs in the NEXT 3 days from t).
    VIX at prediction time is also extracted for regime assignment.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with columns matching `feature_cols` plus 'label' and 'VIX'.
    feature_cols : list of str
        Column names to include in each sequence.
    lookback : int, default=20
        Number of historical days in each input sequence.

    Returns
    -------
    X : np.ndarray, shape (n_valid_samples, lookback, n_features)
        3D tensor of input sequences suitable for GRU input.
    y : np.ndarray, shape (n_valid_samples,)
        Binary labels (1 = drawdown event within 3 days, 0 = no event).
    vix : np.ndarray, shape (n_valid_samples,)
        VIX level at prediction time (day t) for regime classification.
    valid_indices : np.ndarray, shape (n_valid_samples,)
        Original DataFrame indices of valid samples (for date alignment).

    Notes
    -----
    Rows containing ANY NaN value in the sequence window, label, or VIX
    are silently dropped. This is essential because:
    - Early rows may have NaN in lagged features (e.g., ret_lag_5 at row 3)
    - VIX data may have gaps on holidays
    - The label requires 3 future days of data (so last 3 rows are NaN)
    """
    X_list, y_list, vix_list, valid_indices = [], [], [], []

    for i in range(lookback, len(df)):
        # Extract the lookback-day sequence ending just BEFORE the prediction day
        sequence = df[feature_cols].iloc[i-lookback:i].values
        label = df['label'].iloc[i]
        vix_current = df['VIX'].iloc[i]  # VIX at decision point (regime assignment)

        # Skip any sample with NaN anywhere (strict data quality filter)
        if not np.isnan(sequence).any() and not np.isnan(label) and not np.isnan(vix_current):
            X_list.append(sequence)
            y_list.append(label)
            vix_list.append(vix_current)
            valid_indices.append(i)

    return np.array(X_list), np.array(y_list), np.array(vix_list), np.array(valid_indices)


# ============================================================================
# SECTION 3: REGIME-SPECIFIC GRU ARCHITECTURE
# ============================================================================
# Rationale: Financial markets behave differently across volatility regimes.
# In low-VIX environments, drawdowns are rare and driven by sudden shocks.
# In high-VIX environments, drawdowns are more frequent and driven by
# momentum/contagion effects. Training separate models per regime allows
# each GRU to specialise in the dynamics relevant to its regime.

def train_regime_specific_gru(X_train, y_train, VIX_train, config, features):
    """
    Train 3 separate GRU models, one per VIX regime (low/medium/high).

    Each model only sees training data from its own regime, allowing it to
    learn regime-specific temporal patterns. Models use class weighting to
    handle the severe class imbalance (drawdowns are ~13% of samples).

    Parameters
    ----------
    X_train : np.ndarray, shape (n_samples, lookback, n_features)
        Full training sequences (all regimes combined).
    y_train : np.ndarray, shape (n_samples,)
        Binary labels for training data.
    VIX_train : np.ndarray, shape (n_samples,)
        VIX levels at each training sample's prediction time.
    config : dict
        Hyperparameters (hidden_units, dropout, learning_rate, etc.).
    features : list of str
        Feature column names (used for dimensionality reference).

    Returns
    -------
    models : dict
        Mapping of regime name ('low'/'med'/'high') to trained Keras model.
        Regimes with < 100 samples are skipped (insufficient data).
    scalers : dict
        Mapping of regime name to fitted StandardScaler (for inference).

    Notes
    -----
    Each regime's data is independently standardised (zero mean, unit variance)
    using a StandardScaler fit ONLY on that regime's training data. This prevents
    information leakage between regimes during scaling.
    """
    models = {}
    scalers = {}

    # Define VIX regime boundaries (standard market convention):
    # Low: VIX < 15 (calm markets, "complacency")
    # Med: 15 <= VIX < 25 (normal/elevated uncertainty)
    # High: VIX >= 25 (crisis/panic, e.g., COVID March 2020)
    regimes = {
        'low': VIX_train < 15,
        'med': (VIX_train >= 15) & (VIX_train < 25),
        'high': VIX_train >= 25
    }

    for regime_name, mask in regimes.items():
        n_samples = mask.sum()
        event_rate = y_train[mask].mean() if n_samples > 0 else 0

        print(f"  {regime_name} regime: {n_samples} samples, {event_rate:.2%} event rate")

        # Minimum sample threshold: GRUs need sufficient data to learn
        # meaningful temporal patterns. 100 is a conservative lower bound.
        if n_samples < 100:
            print(f"  Skipping {regime_name} regime (insufficient data)")
            continue

        # --- Extract regime-specific training data ---
        X_regime = X_train[mask]
        y_regime = y_train[mask]

        # --- Feature scaling (per-regime) ---
        # StandardScaler normalises each feature to zero mean and unit variance.
        # Applied by reshaping 3D -> 2D, scaling, then reshaping back.
        # This is equivalent to scaling each feature identically across all
        # timesteps (i.e., VIX on day t-20 and day t-1 share the same scale).
        scaler = StandardScaler()
        n_regime, lookback, n_features = X_regime.shape
        X_regime_2d = X_regime.reshape(-1, n_features)          # (n_regime * lookback, n_features)
        X_regime_scaled_2d = scaler.fit_transform(X_regime_2d)  # Fit and transform
        X_regime_scaled = X_regime_scaled_2d.reshape(n_regime, lookback, n_features)  # Back to 3D

        # --- Build GRU model ---
        # Architecture: Single GRU layer -> Dropout -> Sigmoid output
        # - L2 regularisation on output layer to prevent overconfident predictions
        # - recurrent_dropout applies dropout to the recurrent state transitions
        model = Sequential([
            Input(shape=(lookback, n_features)),
            GRU(
                units=config['hidden_units'],
                dropout=config['dropout'],              # Input-to-hidden dropout
                recurrent_dropout=config['dropout'],    # Hidden-to-hidden dropout
                return_sequences=False                  # Only output final hidden state
            ),
            Dropout(config['dropout']),                 # Additional dropout before output
            Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))  # Probability output
        ])

        model.compile(
            optimizer=RMSprop(learning_rate=config['learning_rate']),
            loss='binary_crossentropy',  # Standard loss for probability estimation
            metrics=['accuracy']
        )

        # --- Class weighting for imbalanced data ---
        # Drawdown events (~13%) are much rarer than non-events (~87%).
        # Without class weighting, the model would learn to predict ~0.13 always.
        # 'balanced' mode sets weight inversely proportional to class frequency.
        classes = np.unique(y_regime)
        if len(classes) > 1:
            class_weights_array = compute_class_weight('balanced', classes=classes, y=y_regime)
            class_weights = {int(c): w for c, w in zip(classes, class_weights_array)}
        else:
            class_weights = None  # Degenerate case: only one class present

        # --- Training with early stopping ---
        # EarlyStopping monitors validation loss; if it doesn't improve for
        # `patience` epochs, training stops and best weights are restored.
        # This is the primary defence against overfitting.
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=config['early_stop_patience'],
            restore_best_weights=True,  # Revert to epoch with lowest val_loss
            verbose=0
        )

        model.fit(
            X_regime_scaled, y_regime,
            validation_split=0.2,       # 20% of regime data used for early stopping
            epochs=config['max_epochs'],
            batch_size=config['batch_size'],
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=0                   # Suppress per-epoch output (too verbose for 1620 fits)
        )

        models[regime_name] = model
        scalers[regime_name] = scaler

    return models, scalers


def predict_regime_specific_gru(models, scalers, X_val, VIX_val, config, features):
    """
    Generate probability predictions using the regime-appropriate GRU model.

    For each validation sample, determines its VIX regime, selects the
    corresponding trained model, and generates a drawdown probability.

    Parameters
    ----------
    models : dict
        Trained regime models from train_regime_specific_gru().
    scalers : dict
        Fitted scalers from train_regime_specific_gru().
    X_val : np.ndarray, shape (n_val, lookback, n_features)
        Validation sequences.
    VIX_val : np.ndarray, shape (n_val,)
        VIX levels at validation prediction times.
    config : dict
        Must contain 'lookback' for reshaping.
    features : list of str
        Feature names (unused but kept for API consistency).

    Returns
    -------
    predictions : np.ndarray, shape (n_val,)
        Predicted probabilities of >= 3% drawdown within 3 days.

    Notes
    -----
    Fallback logic: If a sample's regime has no trained model (e.g., no
    high-VIX data in training), it falls back to med -> high -> low.
    If NO models exist, uses 0.13 (approximate unconditional event rate).

    Memory cleanup: After all predictions are made, models are explicitly
    deleted and TensorFlow session is cleared to free GPU memory.
    """
    predictions = np.zeros(len(X_val))

    for i in range(len(X_val)):
        vix = VIX_val[i]

        # --- Regime assignment based on VIX thresholds ---
        if vix < 15:
            regime = 'low'
        elif vix < 25:
            regime = 'med'
        else:
            regime = 'high'

        # --- Fallback: use next-best regime if assigned one has no model ---
        # Priority: med (most data) -> high -> low
        if regime not in models:
            if 'med' in models:
                regime = 'med'
            elif 'high' in models:
                regime = 'high'
            elif 'low' in models:
                regime = 'low'
            else:
                # No models at all -- fall back to unconditional base rate
                predictions[i] = 0.13
                continue

        model = models[regime]
        scaler = scalers[regime]

        # --- Scale single sample using regime's fitted scaler ---
        # Must use transform() (not fit_transform) to use training statistics
        X_sample = X_val[i:i+1]  # Keep 3D shape: (1, lookback, n_features)
        n_features = X_sample.shape[2]
        X_sample_2d = X_sample.reshape(-1, n_features)
        X_sample_scaled_2d = scaler.transform(X_sample_2d)
        X_sample_scaled = X_sample_scaled_2d.reshape(1, config['lookback'], n_features)

        # --- Generate probability prediction ---
        predictions[i] = model.predict(X_sample_scaled, verbose=0)[0, 0]

    # --- Memory cleanup: explicitly free GPU memory after all predictions ---
    for regime_name in list(models.keys()):
        del models[regime_name]
    tf.keras.backend.clear_session()
    gc.collect()

    return predictions


print("Regime-specific GRU helpers defined")


# ============================================================================
# SECTION 3B: ALTERNATIVE MODEL ARCHITECTURES (kept for reference/ablation)
# ============================================================================
# These were tested during development but the regime-specific single-layer
# GRU performed best. Kept here for reproducibility and potential future use.

def build_vanilla_gru(input_shape, config):
    """
    Build a standard single-layer GRU for binary classification.

    Architecture: GRU(hidden_units) -> Dropout -> Dense(1, sigmoid)
    This is the simplest recurrent architecture, serving as the default.

    Parameters
    ----------
    input_shape : tuple
        (lookback, n_features) -- shape of a single input sequence.
    config : dict
        Must contain 'hidden_units', 'dropout', 'learning_rate'.

    Returns
    -------
    model : tf.keras.Model
        Compiled Keras model ready for .fit().
    """
    model = Sequential([
        Input(shape=input_shape),
        GRU(
            units=config['hidden_units'],
            dropout=config['dropout'],
            recurrent_dropout=config['dropout'],
            kernel_regularizer=l2(0.01),    # L2 on input-to-hidden weights
            return_sequences=False          # Output only final timestep
        ),
        Dropout(config['dropout']),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
    ])

    model.compile(
        optimizer=RMSprop(learning_rate=config['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_stacked_gru(input_shape, config):
    """
    Build a 2-layer stacked GRU for binary classification.

    Architecture: GRU(hidden_units, return_sequences=True) -> GRU(hidden_units//2) -> Dropout -> Dense(1)
    The first layer outputs the full sequence of hidden states; the second layer
    compresses this into a single representation. Hypothesis: deeper networks
    may capture more complex temporal patterns.

    Parameters
    ----------
    input_shape : tuple
        (lookback, n_features).
    config : dict
        Must contain 'hidden_units', 'dropout', 'learning_rate'.

    Returns
    -------
    model : tf.keras.Model
        Compiled Keras model ready for .fit().
    """
    model = Sequential([
        Input(shape=input_shape),
        GRU(
            units=config['hidden_units'],
            dropout=config['dropout'],
            recurrent_dropout=config['dropout'],
            kernel_regularizer=l2(0.01),
            return_sequences=True   # Pass full sequence to next GRU layer
        ),
        GRU(
            units=config['hidden_units'] // 2,  # Bottleneck: half the hidden units
            dropout=config['dropout'],
            recurrent_dropout=config['dropout'],
            kernel_regularizer=l2(0.01),
            return_sequences=False  # Output only final hidden state
        ),
        Dropout(config['dropout']),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
    ])

    model.compile(
        optimizer=RMSprop(learning_rate=config['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_bidirectional_gru(input_shape, config):
    """
    Build a Bidirectional GRU for binary classification.

    Architecture: Bidirectional(GRU(hidden_units)) -> Dropout -> Dense(1)
    Processes the sequence in both forward and backward directions, concatenating
    the final hidden states. In theory this captures both causal and anti-causal
    patterns, but may introduce look-ahead bias in time-series contexts.

    Parameters
    ----------
    input_shape : tuple
        (lookback, n_features).
    config : dict
        Must contain 'hidden_units', 'dropout', 'learning_rate'.

    Returns
    -------
    model : tf.keras.Model
        Compiled Keras model ready for .fit().

    Notes
    -----
    The output dimension is 2 * hidden_units due to the forward/backward
    concatenation. This doubles the parameters in the Dense layer.
    """
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(
            GRU(
                units=config['hidden_units'],
                dropout=config['dropout'],
                recurrent_dropout=config['dropout'],
                kernel_regularizer=l2(0.01),
                return_sequences=False
            )
        ),
        Dropout(config['dropout']),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
    ])

    model.compile(
        optimizer=RMSprop(learning_rate=config['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================================
# SECTION 4: WALK-FORWARD TRAINING LOOP
# =========================================================================