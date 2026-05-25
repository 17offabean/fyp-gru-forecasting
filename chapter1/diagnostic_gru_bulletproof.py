#!/usr/bin/env python3
"""
diagnostic_gru_bulletproof.py
=============================
Minimal 3-Feature GRU for NASDAQ-100 Drawdown Prediction -- MAIN THESIS MODEL.

Thesis: "Probabilistic Risk Estimation in Financial Markets using GRU Neural Networks"
Chapter 1: NASDAQ-100 Drawdown Prediction

Purpose:
    This is the PRIMARY model script for the thesis. It trains a minimal single-layer
    GRU using only 3 features (log_ret, VIX, realized_vol_5d) to predict the probability
    of a >= 3% drawdown in the NASDAQ-100 index within the next 3 trading days.

    The key thesis argument is FEATURE PARSIMONY: this minimal model should outperform
    the full 12-feature model (05_GRU_architecture_bulletproof.py), demonstrating that
    carefully selected features beat the kitchen-sink approach for financial forecasting.

Methodology:
    1. Walk-Forward Validation: 756-day train / 252-day val / 252-day step.
       23 total folds split 80/20 into tuning (18) and held-out (5) folds.
    2. Baseline Comparison: GRU vs constant predictor vs logistic regression
       on tuning folds.
    3. Hyperparameter Search: 20 random configurations evaluated on tuning folds.
    4. Held-Out Evaluation: Best config retrained and evaluated on 5 held-out folds.
    5. Monte Carlo Permutation Test (MCPT): 200 permutations per held-out fold
       to establish statistical significance (p-value) of the GRU's performance.
    6. Visualisation: 6-panel dashboard summarising all results.

Inputs:
    - data/features_^NDX.parquet: Feature matrix with 'label' column.
    - results/baselines_^NDX.parquet: Pre-computed baseline Brier scores
      (constant predictor and logistic regression, from a prior script).

Outputs:
    - results/ch1/figures/tuning_baseline_comparison.{svg,png}
    - results/ch1/figures/hyperparam_search_scatter.{svg,png}
    - results/ch1/figures/hyperparam_search_table.{svg,png}
    - results/ch1/figures/holdout_brier_comparison.{svg,png}
    - results/ch1/figures/holdout_improvement_pct.{svg,png}
    - results/ch1/figures/mcpt_null_distribution_overall.{svg,png}
    - results/ch1/figures/mcpt_null_per_fold.{svg,png}
    - results/ch1/figures/gru_final_evaluation_dashboard.{svg,png}
    - results/ch1/tables/tuning_brier_scores.csv
    - results/ch1/tables/hyperparam_search_results.csv
    - results/ch1/tables/holdout_results.csv
    - results/ch1/tables/mcpt_results.csv
    - results/ch1/best_config.json
    - results/ch1/diagnostic_gru_summary.json
    - results/ch1/mcpt_checkpoint.json (intermediate, for crash recovery)

Runtime:
    8-10 hours (dominated by MCPT: 200 permutations x 5 folds = 1000 model fits).
    Designed for overnight execution with checkpoint/resume capability.

Key Design Decisions:
    - No feature scaling: The 3 minimal features are already on comparable scales,
      and scaling was found to not improve performance in preliminary experiments.
    - No class weighting: Unlike the regime-specific model, this simpler model
      benefits from learning the natural class distribution.
    - MCPT checkpoint/resume: If the script crashes mid-MCPT (e.g., OOM on fold 3),
      it saves progress and can resume from the last completed fold on re-run.

Author: Junior Kinyanjui
DO NOT RUN IN CI -- designed for overnight local execution (8-10 hours).
"""

# ============================================================================
# SECTION 0: ENVIRONMENT SETUP AND CONFIGURATION
# ============================================================================
# Standard library imports
import gc          # Garbage collection for explicit memory management
import json        # JSON serialisation for config/results files
import os          # File system operations
import random      # Python random number generator (seeded for reproducibility)
import time        # Wall-clock timing for runtime estimation

# Scientific computing stack
import numpy as np
import pandas as pd
from pathlib import Path

# TensorFlow/Keras for GRU model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

# Matplotlib for figure generation
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend: no display required (server/overnight use)
import matplotlib.pyplot as plt

# ============================================================================
# REPRODUCIBILITY SEEDS
# Setting seeds for all three random number generators used in the pipeline.
# This ensures the hyperparameter search samples the same configs, weight
# initialisation is deterministic, and permutation shuffles are reproducible.
# ============================================================================
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- Thesis colour scheme (consistent across all Chapter 1 figures) ---
CLR_GRU      = '#4682B4'   # steelblue      - GRU model results
CLR_CONSTANT = '#FF7F50'   # coral           - constant (naive) baseline
CLR_LOGISTIC = '#FFD700'   # gold            - logistic regression baseline
CLR_NULL     = '#B0C4DE'   # lightsteelblue  - MCPT null distribution

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

# --- Output directories ---
FIG_DIR = os.path.join('results', 'ch1', 'figures')
TBL_DIR = os.path.join('results', 'ch1', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)
os.makedirs('results/ch1', exist_ok=True)


def save_thesis_fig(name, fig=None):
    """
    Save a matplotlib figure in dual format for thesis inclusion.

    Produces both SVG (vector format for LaTeX/Word) and high-DPI PNG
    (raster format for compatibility/preview) in the standard output directory.

    Parameters
    ----------
    name : str
        Base filename without extension (e.g., 'mcpt_null_distribution_overall').
    fig : matplotlib.figure.Figure, optional
        Figure object to save. Defaults to current active figure.
    """
    if fig is None:
        fig = plt.gcf()
    fig.savefig(os.path.join(FIG_DIR, f'{name}.svg'), format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {name}.svg + {name}.png")


# ============================================================================
# SECTION 1: DATA LOADING AND WALK-FORWARD SPLIT DEFINITION
# ============================================================================

# --- Load pre-computed feature matrix ---
DATA_DIR = Path("data")
df = pd.read_parquet(DATA_DIR / "features_^NDX.parquet")


def walkforward_splits(n, train_days=756, val_days=252, step_days=252):
    """
    Generate walk-forward (rolling window) train/validation split indices.

    Implements a time-series cross-validation scheme that strictly respects
    temporal ordering -- training data always precedes validation data, preventing
    look-ahead bias (future information leaking into training).

    The scheme uses FIXED-SIZE windows (not expanding), meaning each fold trains
    on exactly 756 days (~3 years) and validates on 252 days (~1 year).

    Parameters
    ----------
    n : int
        Total number of trading days in the dataset.
    train_days : int, default=756
        Number of training days per fold (~3 calendar years).
    val_days : int, default=252
        Number of validation days per fold (~1 calendar year).
    step_days : int, default=252
        How far to advance the window between folds (~1 year).

    Returns
    -------
    list of dict
        Each dict contains integer indices:
        - 'train_start': First day of training window
        - 'train_end':   Last day of training window (exclusive)
        - 'val_start':   First day of validation window
        - 'val_end':     Last day of validation window (exclusive)

    Example
    -------
    For n=6000:
        Fold 0:  train=[0, 756),      val=[756, 1008)
        Fold 1:  train=[252, 1008),   val=[1008, 1260)
        Fold 2:  train=[504, 1260),   val=[1260, 1512)
        ... (23 folds total)
    """
    splits, start = [], 0
    while True:
        train_end = start + train_days
        val_end   = train_end + val_days
        if val_end > n:
            break
        splits.append({"train_start": start, "train_end": train_end,
                        "val_start": train_end, "val_end": val_end})
        start += step_days
    return splits


splits = walkforward_splits(len(df))
print(f"Total walk-forward folds: {len(splits)}")

# Print event rates per fold to check for temporal non-stationarity
# (if event rates vary wildly, the prediction task is harder)
for i, s in enumerate(splits):
    tr = df.iloc[s['train_start']:s['train_end']]['label'].mean()
    vl = df.iloc[s['val_start']:s['val_end']]['label'].mean()
    print(f"  Fold {i:2d}: train_rate={tr:.3f}  val_rate={vl:.3f}")

# --- 80/20 split of folds into tuning vs held-out ---
# Tuning folds: used for hyperparameter search (configs are selected based on these)
# Held-out folds: NEVER seen during tuning; used only for final unbiased evaluation
# This two-level split prevents overfitting to the validation folds during HP search.
n_tuning       = int(len(splits) * 0.8)
TUNING_FOLDS   = list(range(n_tuning))
HELD_OUT_FOLDS = list(range(n_tuning, len(splits)))
print(f"\nTuning folds  ({len(TUNING_FOLDS)}/{len(splits)}, 80%): {TUNING_FOLDS}")
print(f"Held-out folds ({len(HELD_OUT_FOLDS)}/{len(splits)}, 20%): {HELD_OUT_FOLDS}")


# ============================================================================
# SECTION 1B: SEQUENCE CREATION FUNCTION
# ============================================================================

def create_sequences(df_chunk, feature_cols, lookback=10):
    """
    Convert a DataFrame slice into 3D input tensors for the GRU.

    For each day t (starting from index `lookback`), extracts a sequence of
    `lookback` consecutive days [t-lookback, t) as input, and the binary label
    at day t as the target. Samples with ANY NaN value are discarded.

    Parameters
    ----------
    df_chunk : pd.DataFrame
        A slice of the full feature DataFrame (e.g., training or validation portion).
    feature_cols : list of str
        Column names to include in the sequence (e.g., ['log_ret', 'VIX', ...]).
    lookback : int, default=10
        Number of historical trading days in each input sequence.

    Returns
    -------
    X : np.ndarray, shape (n_valid_samples, lookback, n_features)
        3D tensor of input sequences. Each sample is a (lookback x n_features) matrix
        representing the recent history at the time of prediction.
    y : np.ndarray, shape (n_valid_samples,)
        Binary labels: 1 if >= 3% drawdown occurs within next 3 days, 0 otherwise.

    Notes
    -----
    - NaN filtering is CRITICAL: early rows have NaN in lagged features,
      and the last 3 rows have NaN labels (label requires future data).
    - No VIX regime assignment here (unlike 05_GRU_architecture_bulletproof.py)
      because this model does NOT use regime-specific training.
    - The lookback window does NOT overlap with the prediction horizon:
      sequence = days [t-10, t), prediction = whether drawdown in [t, t+3).
    """
    X_list, y_list = [], []
    for i in range(lookback, len(df_chunk)):
        seq   = df_chunk[feature_cols].iloc[i - lookback:i].values  # Shape: (lookback, n_features)
        label = df_chunk['label'].iloc[i]
        # Strict NaN filter: skip if ANY value in the sequence or label is NaN
        if not np.isnan(seq).any() and not np.isnan(label):
            X_list.append(seq)
            y_list.append(label)
    return np.array(X_list), np.array(y_list)


# ============================================================================
# SECTION 1C: FEATURE SET AND BASELINE CONFIGURATION
# ============================================================================

# --- Minimal feature set (3 features) ---
# These were selected through domain knowledge and ablation studies:
# - log_ret:          Captures momentum/mean-reversion at daily frequency
# - VIX:             Implied volatility (forward-looking fear gauge)
# - realized_vol_5d: Backward-looking volatility (recent turbulence)
# Together these capture the core risk dynamics with minimal redundancy.
minimal_features = ['log_ret', 'VIX', 'realized_vol_5d']

# --- Default model configuration (pre-tuning) ---
# Conservative defaults chosen to avoid overfitting on financial data:
# - Short lookback (10 days): Financial signals decay rapidly
# - Small hidden units (16): Limits model capacity for small datasets
# - No dropout (0.0): Baseline without regularisation to assess raw capacity
minimal_config = {
    'lookback':      10,     # 10 trading days (~2 weeks)
    'hidden_units':  16,     # Small GRU state (prevents memorisation)
    'dropout':       0.0,    # No dropout for baseline (added during HP search)
    'learning_rate': 0.001,  # Standard RMSprop learning rate
    'batch_size':    32,     # Standard mini-batch size
    'max_epochs':    50,     # Moderate epoch budget (early stopping usually fires earlier)
}


# ============================================================================
# SECTION 1D: MODEL BUILDER AND TRAINING HELPER
# ============================================================================

def build_gru_model(input_shape, config):
    """
    Build and compile a single-layer GRU for binary classification.

    Architecture: Input -> GRU(hidden_units) -> Dense(1, sigmoid)
    This is intentionally simple -- the thesis argues that model complexity
    is less important than feature quality for financial forecasting.

    Parameters
    ----------
    input_shape : tuple
        (lookback, n_features) -- dimensions of a single input sequence.
    config : dict
        Must contain: 'hidden_units', 'dropout', 'learning_rate'.

    Returns
    -------
    model : tf.keras.Model
        Compiled model ready for .fit(). Uses:
        - RMSprop optimiser (adaptive learning rates, good for RNNs)
        - Binary cross-entropy loss (proper scoring rule for probabilities)
        - Accuracy metric (for monitoring, not for model selection)

    Notes
    -----
    The GRU forward pass for a sequence [x_1, ..., x_T]:
        For each timestep t:
            z_t = sigmoid(W_z * x_t + U_z * h_{t-1})        # Update gate
            r_t = sigmoid(W_r * x_t + U_r * h_{t-1})        # Reset gate
            h_t = (1-z_t) * h_{t-1} + z_t * tanh(W * x_t + U * (r_t . h_{t-1}))
        Output: h_T (final hidden state) -> Dense(1, sigmoid) -> P(drawdown)

    The update gate z_t controls how much of the past to keep vs. update.
    The reset gate r_t controls how much of the past to expose to the candidate.
    """
    model = Sequential([
        Input(shape=input_shape),
        GRU(units=config['hidden_units'],
            dropout=config['dropout'],           # Input-to-hidden dropout (applied at each timestep)
            recurrent_dropout=config['dropout'],  # Hidden-to-hidden dropout (regularises temporal dynamics)
            return_sequences=False),              # Only output h_T (final hidden state)
        Dense(1, activation='sigmoid'),          # Sigmoid maps to [0, 1] probability
    ])
    model.compile(optimizer=RMSprop(learning_rate=config['learning_rate']),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def run_config_on_folds(config, feature_cols, folds):
    """
    Evaluate a single hyperparameter configuration across multiple folds.

    Trains and evaluates the GRU on each specified fold, collecting per-fold
    Brier scores. This is the inner loop of the hyperparameter search.

    Parameters
    ----------
    config : dict
        Hyperparameters to test (lookback, hidden_units, dropout, etc.).
    feature_cols : list of str
        Feature columns for sequence creation.
    folds : list of int
        Fold indices to evaluate on.

    Returns
    -------
    mean_brier : float
        Mean Brier score across all valid folds.
    std_brier : float
        Standard deviation of per-fold Brier scores.
    brier_scores : list of float
        Individual Brier score for each fold (for variance analysis).

    Notes
    -----
    - Folds with < 100 training or < 10 validation samples are skipped.
    - Memory is explicitly freed after each fold (del model + clear_session).
    - The Brier score is computed as: mean( (p_hat - y)^2 ) where p_hat is
      the predicted probability and y is the actual binary outcome.
    """
    brier_scores = []
    for fold_idx in folds:
        split    = splits[fold_idx]
        df_train = df.iloc[split['train_start']:split['train_end']]
        df_val   = df.iloc[split['val_start']:split['val_end']]

        X_train, y_train = create_sequences(df_train, feature_cols, config['lookback'])
        X_val,   y_val   = create_sequences(df_val,   feature_cols, config['lookback'])

        # Skip folds with insufficient data after NaN filtering
        if len(X_train) < 100 or len(X_val) < 10:
            print(f"Fold {fold_idx}: SKIP (too few samples)")
            continue

        model = build_gru_model((config['lookback'], len(feature_cols)), config)
        model.fit(X_train, y_train, validation_split=0.2,
                  epochs=config['max_epochs'], batch_size=config['batch_size'], verbose=0)

        # Generate probability predictions on validation set
        val_pred = model.predict(X_val, verbose=0).flatten()

        # Brier score: mean squared error between predicted probabilities and actual outcomes
        # Lower is better. Perfect = 0.0, predicting base rate ~= 0.0937
        brier    = np.mean((val_pred - y_val) ** 2)
        brier_scores.append(brier)
        print(f"Fold {fold_idx}: Brier = {brier:.6f}")

        # Explicit memory cleanup to prevent GPU OOM across many folds
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    if not brier_scores:
        return np.nan, np.nan, []
    return float(np.mean(brier_scores)), float(np.std(brier_scores)), brier_scores


# ============================================================================
# SECTION 2: BASELINE COMPARISON ON TUNING FOLDS
# ============================================================================
# Compare the GRU against two baselines on the tuning folds:
# 1. Constant predictor: Always predicts P(drawdown) = training event rate
# 2. Logistic regression: Linear model using the same features
# Both baselines were pre-computed by a separate script and saved to parquet.
print("\n" + "=" * 60)
print("SECTION 2: BASELINE COMPARISON (Tuning Folds)")
print("=" * 60)

try:
    # --- Load pre-computed baseline Brier scores ---
    # These come from a prior script that ran constant and logistic models
    # on ALL folds. We subset to tuning folds only.
    baselines_raw = pd.read_parquet(Path("results") / "baselines_^NDX.parquet")
    fold_brier = (baselines_raw
                  .groupby(['split_index', 'model_name'])['brier_score']
                  .mean()
                  .unstack('model_name'))
    tuning_brier = fold_brier.loc[TUNING_FOLDS]

    print("Pre-computed baseline Brier scores (tuning folds):")
    print(tuning_brier.to_string())

    # --- Run minimal GRU on each tuning fold ---
    gru_results = []
    histories   = []  # Store training histories for loss curve plotting

    for fold_idx in TUNING_FOLDS:
        split    = splits[fold_idx]
        df_train = df.iloc[split['train_start']:split['train_end']]
        df_val   = df.iloc[split['val_start']:split['val_end']]

        X_train, y_train = create_sequences(df_train, minimal_features, minimal_config['lookback'])
        X_val,   y_val   = create_sequences(df_val,   minimal_features, minimal_config['lookback'])

        # Build and train a fresh model for this fold
        model = Sequential([
            Input(shape=(minimal_config['lookback'], len(minimal_features))),
            GRU(units=minimal_config['hidden_units'], return_sequences=False),
            Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer=RMSprop(learning_rate=minimal_config['learning_rate']),
                      loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, validation_split=0.2,
                            epochs=minimal_config['max_epochs'],
                            batch_size=minimal_config['batch_size'], verbose=0)

        # Evaluate on validation fold
        gru_pred  = model.predict(X_val, verbose=0).flatten()
        brier_gru = float(np.mean((gru_pred - y_val) ** 2))

        histories.append(history.history)
        gru_results.append({'fold': fold_idx, 'brier_gru': brier_gru})

        # Print comparison with baselines for this fold
        b_naive = tuning_brier.loc[fold_idx, 'constant']
        b_lr    = tuning_brier.loc[fold_idx, 'logistic']
        print(f"\nFold {fold_idx}  | Naive={b_naive:.6f}  LR={b_lr:.6f}  GRU={brier_gru:.6f}")

        # Memory cleanup
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    # --- Combine GRU results with baseline results ---
    gru_df     = pd.DataFrame(gru_results).set_index('fold')
    baseline_df = tuning_brier.join(gru_df)

    # Brier Skill Score (BSS): Measures improvement relative to a reference model.
    # BSS = 1 - (Brier_model / Brier_reference)
    # BSS > 0 means the model outperforms the reference; BSS = 1 is perfect.
    baseline_df['bss_gru'] = 1 - baseline_df['brier_gru'] / baseline_df['constant']
    baseline_df['bss_lr']  = 1 - baseline_df['logistic']   / baseline_df['constant']

    # Save tuning brier scores as CSV for thesis table
    baseline_df.to_csv(os.path.join(TBL_DIR, 'tuning_brier_scores.csv'))
    print(f"\nSaved: tuning_brier_scores.csv")

    # --- 3-panel comparison figure ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Panel 1: GRU training/validation loss curves (mean +/- std across folds)
    # Shows whether the model converges and whether it overfits.
    ax = axes[0]
    max_ep = max(len(h['loss']) for h in histories)
    # Pad shorter histories to align for averaging (repeat final value)
    pad    = lambda lst: lst + [lst[-1]] * (max_ep - len(lst))
    tl     = np.array([pad(h['loss'])     for h in histories])
    vl     = np.array([pad(h['val_loss']) for h in histories])
    ep     = range(1, max_ep + 1)
    ax.plot(ep, tl.mean(0), color=CLR_GRU, linestyle='-', label='Train (mean)', lw=2)
    ax.fill_between(ep, tl.mean(0) - tl.std(0), tl.mean(0) + tl.std(0),
                    alpha=0.2, color=CLR_GRU)
    ax.plot(ep, vl.mean(0), color=CLR_CONSTANT, linestyle='--', label='Val (mean)', lw=2)
    ax.fill_between(ep, vl.mean(0) - vl.std(0), vl.mean(0) + vl.std(0),
                    alpha=0.2, color=CLR_CONSTANT)
    ax.set_xlabel('Epoch'); ax.set_ylabel('BCE Loss')
    ax.set_title('GRU Loss (mean +/- std, tuning folds)'); ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 2: Brier scores by fold (grouped bar chart: Naive vs LR vs GRU)
    ax = axes[1]
    x, w = np.arange(len(TUNING_FOLDS)), 0.25
    ax.bar(x - w, baseline_df['constant'],  w, label='Naive',    alpha=0.8, color=CLR_CONSTANT)
    ax.bar(x,     baseline_df['logistic'],  w, label='Logistic', alpha=0.8, color=CLR_LOGISTIC)
    ax.bar(x + w, baseline_df['brier_gru'], w, label='GRU',      alpha=0.8, color=CLR_GRU)
    ax.set_xlabel('Fold'); ax.set_ylabel('Brier Score')
    ax.set_title('Brier Score by Fold'); ax.set_xticks(x)
    ax.set_xticklabels(TUNING_FOLDS); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Brier Skill Score vs Naive baseline
    # Positive BSS = model improves on naive; negative = worse than naive
    ax = axes[2]
    ax.bar(x - w/2, baseline_df['bss_gru'], w, label='GRU',
           alpha=0.8, color=[CLR_GRU if v > 0 else 'red' for v in baseline_df['bss_gru']])
    ax.bar(x + w/2, baseline_df['bss_lr'],  w, label='Logistic',
           alpha=0.8, color=[CLR_LOGISTIC if v > 0 else 'orange' for v in baseline_df['bss_lr']])
    ax.axhline(0, color='black', lw=1.5)  # Zero line: performance equal to naive
    ax.set_xlabel('Fold'); ax.set_ylabel('Brier Skill Score (vs Naive)')
    ax.set_title('Skill vs Naive Baseline'); ax.set_xticks(x)
    ax.set_xticklabels(TUNING_FOLDS); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_thesis_fig('tuning_baseline_comparison', fig)
    plt.close(fig)

    # --- Print summary statistics ---
    print(f"\n{'='*60}")
    print("SUMMARY (Tuning Folds)")
    print(f"{'='*60}")
    print(baseline_df[['constant', 'logistic', 'brier_gru', 'bss_gru', 'bss_lr']].to_string())
    print(f"\nMean Naive Brier : {baseline_df['constant'].mean():.6f}")
    print(f"Mean LR Brier    : {baseline_df['logistic'].mean():.6f}")
    print(f"Mean GRU Brier   : {baseline_df['brier_gru'].mean():.6f}")
    print(f"Mean GRU BSS vs Naive : {baseline_df['bss_gru'].mean():+.4f}")
    print(f"Mean LR  BSS vs Naive : {baseline_df['bss_lr'].mean():+.4f}")

except Exception as e:
    print(f"\n[ERROR] Section 2 failed: {e}")
    import traceback; traceback.print_exc()
    raise


# ============================================================================
# SECTION 3: HYPERPARAMETER SEARCH (Random Search on Tuning Folds)
# ============================================================================
# Test 20 random configurations to find the best GRU hyperparameters.
# Only tuning folds are used -- held-out folds remain untouched.
# This prevents "double-dipping" where HP selection would overfit to test data.
print("\n" + "=" * 60)
print("SECTION 3: HYPERPARAMETER SEARCH (20 configs, Tuning Folds)")
print("=" * 60)

try:
    # --- Define search space ---
    # Ranges chosen based on financial ML best practices and preliminary experiments:
    # - Short lookbacks (10-20): Financial signals decay within weeks
    # - Small hidden units (16-48): Prevents overfitting on noisy financial data
    # - Moderate dropout (0-0.3): Regularisation for generalisation
    SEARCH_SPACE = {
        'lookback':      [10, 15, 20],          # Sequence length (trading days)
        'hidden_units':  [16, 32, 48],          # GRU state dimensionality
        'dropout':       [0.0, 0.2, 0.3],       # Dropout rate
        'learning_rate': [0.0005, 0.001],       # RMSprop learning rate
        'batch_size':    [32, 64],              # Mini-batch size
    }

    N_CONFIGS = 20  # Number of random samples from search space

    # --- Generate random configurations (deterministic due to seed=42) ---
    configs_to_test = []
    for _ in range(N_CONFIGS):
        cfg = {
            'lookback':      random.choice(SEARCH_SPACE['lookback']),
            'hidden_units':  random.choice(SEARCH_SPACE['hidden_units']),
            'dropout':       random.choice(SEARCH_SPACE['dropout']),
            'learning_rate': random.choice(SEARCH_SPACE['learning_rate']),
            'batch_size':    random.choice(SEARCH_SPACE['batch_size']),
            'max_epochs':    50,
        }
        configs_to_test.append(cfg)

    # --- Evaluate each configuration ---
    all_results = []

    for idx, cfg in enumerate(configs_to_test):
        print(f"\nConfig {idx + 1}/{len(configs_to_test)}: {cfg}")
        mean_brier, std_brier, per_fold = run_config_on_folds(
            cfg,
            feature_cols=minimal_features,
            folds=TUNING_FOLDS,
        )
        all_results.append({
            'config_idx':     idx,
            'config':         cfg,
            'mean_brier':     mean_brier,
            'std_brier':      std_brier,
            'brier_per_fold': per_fold,
        })
        print(f"  => Mean Brier: {mean_brier:.6f}, Std: {std_brier:.6f}")

    # Sort by mean Brier (ascending = best first)
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('mean_brier').reset_index(drop=True)

    # --- Save hyperparameter search results ---
    search_csv = results_df[['config_idx', 'mean_brier', 'std_brier']].copy()
    for key in ['lookback', 'hidden_units', 'dropout', 'learning_rate', 'batch_size']:
        search_csv[key] = results_df['config'].apply(lambda c: c[key])
    search_csv.to_csv(os.path.join(TBL_DIR, 'hyperparam_search_results.csv'), index=False)
    print(f"\nSaved: hyperparam_search_results.csv")

    # --- Select best configuration ---
    if not results_df.empty and not pd.isna(results_df.iloc[0]['mean_brier']):
        best_config = results_df.iloc[0]['config']
        print(f"Best config from tuning search: {best_config}")
    else:
        # Fallback if all configs failed (should never happen in practice)
        best_config = {'lookback': 20, 'hidden_units': 32, 'dropout': 0.3,
                       'learning_rate': 0.001, 'batch_size': 32, 'max_epochs': 50}
        print(f"Using fallback best config: {best_config}")

    # Save best config as JSON for reproducibility
    with open('results/ch1/best_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    print("Saved: best_config.json")

    # --- Figure: Scatter plot of all search results ---
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_search = results_df.dropna(subset=['mean_brier']).reset_index(drop=True)
    ax.scatter(range(len(valid_search)), valid_search['mean_brier'],
               alpha=0.7, color=CLR_GRU, s=60, edgecolors='black', linewidth=0.5)
    ax.axhline(valid_search['mean_brier'].min(), color='red', linestyle='--',
               label=f"Best: {valid_search['mean_brier'].min():.4f}")
    ax.set_xlabel('Config Index (sorted by Brier)')
    ax.set_ylabel('Mean Brier Score')
    ax.set_title('Hyperparameter Search Results (Tuning Folds)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_thesis_fig('hyperparam_search_scatter', fig)
    plt.close(fig)

    # --- Figure: Top-10 configs as a formatted table ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    table_data = []
    for _, row in valid_search.head(10).iterrows():
        c = row['config']
        table_data.append([
            f"{row['config_idx']:.0f}",
            f"{c['lookback']}",
            f"{c['hidden_units']}",
            f"{c['dropout']:.1f}",
            f"{c['learning_rate']}",
            f"{c['batch_size']}",
            f"{row['mean_brier']:.6f}",
            f"{row['std_brier']:.6f}",
        ])
    col_labels = ['Idx', 'Lookback', 'Hidden', 'Dropout', 'LR', 'Batch', 'Mean Brier', 'Std Brier']
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    # Highlight best row in green
    for j in range(len(col_labels)):
        table[1, j].set_facecolor('#d4edda')
    ax.set_title('Top 10 Hyperparameter Configurations (sorted by Mean Brier)', fontsize=13, pad=20)
    plt.tight_layout()
    save_thesis_fig('hyperparam_search_table', fig)
    plt.close(fig)

except Exception as e:
    print(f"\n[ERROR] Section 3 failed: {e}")
    import traceback; traceback.print_exc()
    raise


# ============================================================================
# SECTION 4: HELD-OUT EVALUATION (Final Unbiased Assessment)
# ============================================================================
# Train the best hyperparameter configuration on each held-out fold and
# evaluate against baselines. This is the DEFINITIVE performance measurement
# reported in the thesis, because these folds were never used for any
# model selection decisions.
print("\n" + "=" * 60)
print("SECTION 4: HELD-OUT EVALUATION (Best Config)")
print("=" * 60)

try:
    print(f"Best config: {best_config}")
    print(f"Held-out folds: {HELD_OUT_FOLDS}")

    # Get baseline Brier scores for held-out folds
    held_brier = fold_brier.loc[fold_brier.index.intersection(HELD_OUT_FOLDS)]

    # --- Run tuned GRU on each held-out fold ---
    print("\nRunning tuned GRU on held-out folds...")
    final_results = []

    for fold_idx in held_brier.index:
        split    = splits[fold_idx]
        df_train = df.iloc[split['train_start']:split['train_end']]
        df_val   = df.iloc[split['val_start']:split['val_end']]

        X_train, y_train = create_sequences(df_train, minimal_features, best_config['lookback'])
        X_val,   y_val   = create_sequences(df_val,   minimal_features, best_config['lookback'])

        if len(X_train) < 100 or len(X_val) < 10:
            continue

        # Use early stopping on held-out evaluation too (prevents overfitting)
        early_stop = EarlyStopping(monitor='val_loss', patience=5,
                                   restore_best_weights=True, verbose=0)
        model = build_gru_model((best_config['lookback'], len(minimal_features)), best_config)
        model.fit(X_train, y_train,
                  validation_split=0.2,
                  epochs=best_config['max_epochs'],
                  batch_size=best_config['batch_size'],
                  callbacks=[early_stop],
                  verbose=0)
        gru_pred  = model.predict(X_val, verbose=0).flatten()
        brier_gru = float(np.mean((gru_pred - y_val) ** 2))

        # Compare against baselines
        b_naive = held_brier.loc[fold_idx, 'constant']
        b_lr    = held_brier.loc[fold_idx, 'logistic']
        bss     = 1 - brier_gru / b_lr  # BSS relative to logistic regression
        imp_pct = (b_lr - brier_gru) / b_lr * 100  # Percentage improvement vs LR
        print(f"Fold {fold_idx}: Naive={b_naive:.4f}  LR={b_lr:.4f}  "
              f"GRU={brier_gru:.4f}  BSS_vs_LR={bss:.4f}  Imp_vs_LR={imp_pct:+.1f}%")

        final_results.append({
            'fold':             fold_idx,
            'brier_naive':      b_naive,
            'brier_lr':         b_lr,
            'brier_gru':        brier_gru,
            'bss':              bss,
            'improvement_pct':  imp_pct,
            'train_event_rate': float(y_train.mean()),  # For stationarity analysis
            'val_event_rate':   float(y_val.mean()),    # For stationarity analysis
        })

        del model
        tf.keras.backend.clear_session()
        gc.collect()

    final_df = pd.DataFrame(final_results)

    # --- Compute summary statistics (used in later sections and dashboard) ---
    holdout_mean           = final_df['brier_gru'].mean()
    mean_lr                = final_df['brier_lr'].mean()
    mean_constant          = final_df['brier_naive'].mean()
    overall_improvement    = final_df['improvement_pct'].mean()
    folds_beating_baseline = int((final_df['bss'] > 0).sum())
    valid_folds            = final_df['fold'].tolist()

    # Save holdout results
    final_df.to_csv(os.path.join(TBL_DIR, 'holdout_results.csv'), index=False)
    print(f"\nSaved: holdout_results.csv")

    print(f"\n{'='*60}")
    print("FINAL EVALUATION SUMMARY (Held-Out Folds) -- vs Logistic Regression")
    print(f"{'='*60}")
    print(final_df[['fold', 'brier_naive', 'brier_lr', 'brier_gru', 'bss',
                    'improvement_pct']].to_string(index=False))
    print(f"\nMean Naive Brier : {mean_constant:.6f}")
    print(f"Mean LR Brier    : {mean_lr:.6f}")
    print(f"Mean GRU Brier   : {holdout_mean:.6f}")
    print(f"Mean Improvement vs LR : {overall_improvement:+.2f}%")
    print(f"Mean BSS (GRU vs LR)   : {final_df['bss'].mean():+.4f}")
    print(f"Folds GRU beats LR     : {folds_beating_baseline}/{len(final_df)}")

    # --- Figure: Held-out Brier comparison (grouped bar chart) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x, w = np.arange(len(final_df)), 0.25
    ax.bar(x - w, final_df['brier_naive'], w, label='Naive',    alpha=0.8, color=CLR_CONSTANT)
    ax.bar(x,     final_df['brier_lr'],    w, label='Logistic', alpha=0.8, color=CLR_LOGISTIC)
    ax.bar(x + w, final_df['brier_gru'],   w, label='GRU',      alpha=0.8, color=CLR_GRU)
    ax.set_xlabel('Fold'); ax.set_ylabel('Brier Score')
    ax.set_title('Brier Score -- Held-Out Folds'); ax.set_xticks(x)
    ax.set_xticklabels(final_df['fold'].tolist()); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_thesis_fig('holdout_brier_comparison', fig)
    plt.close(fig)

    # --- Figure: Improvement percentage vs logistic regression ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if v > 0 else 'red' for v in final_df['improvement_pct']]
    ax.bar(x, final_df['improvement_pct'], alpha=0.8, color=colors)
    ax.axhline(0, color='black', linewidth=1.5)
    ax.axhline(overall_improvement, color=CLR_GRU, linewidth=2, linestyle='--',
               label=f'Mean: {overall_improvement:.1f}%')
    ax.set_xlabel('Fold'); ax.set_ylabel('Improvement (%)')
    ax.set_title('GRU Improvement vs Logistic Regression (Held-Out)')
    ax.set_xticks(x); ax.set_xticklabels(final_df['fold'].tolist())
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_thesis_fig('holdout_improvement_pct', fig)
    plt.close(fig)

except Exception as e:
    print(f"\n[ERROR] Section 4 failed: {e}")
    import traceback; traceback.print_exc()
    raise


# ============================================================================
# SECTION 5: MONTE CARLO PERMUTATION TEST (MCPT)
# ============================================================================
# The MCPT establishes whether the GRU's performance is statistically significant
# or could have arisen by chance. It works by:
# 1. Training the REAL model and recording its Brier score on held-out folds.
# 2. Repeating 200 times with SHUFFLED training labels (destroying any real signal).
# 3. Computing a p-value: proportion of null models that beat or match the real model.
#
# If p < 0.05, we reject the null hypothesis that "the GRU learned nothing meaningful"
# at the 5% significance level.
#
# This is more robust than parametric tests because it makes no assumptions about
# the distribution of Brier scores. It directly answers: "How likely is this
# performance if the features contained no predictive information?"
print("\n" + "=" * 60)
print("SECTION 5: MONTE CARLO PERMUTATION TEST (MCPT)")
print("=" * 60)

try:
    MCPT_PERMUTATIONS_PER_FOLD = 200  # 200 null models per held-out fold
    print(f"Permutations per fold: {MCPT_PERMUTATIONS_PER_FOLD}")
    print(f"Held-out folds:        {len(HELD_OUT_FOLDS)}")
    print(f"Total null samples:    {MCPT_PERMUTATIONS_PER_FOLD * len(HELD_OUT_FOLDS)}")
    print(f"\nEstimated runtime: "
          f"{MCPT_PERMUTATIONS_PER_FOLD * len(HELD_OUT_FOLDS) * 15 / 3600:.1f}--"
          f"{MCPT_PERMUTATIONS_PER_FOLD * len(HELD_OUT_FOLDS) * 30 / 3600:.1f} hours\n")

    # --- CHECKPOINT/RESUME LOGIC ---
    # The MCPT takes 8+ hours. If the script crashes (e.g., OOM on fold 3),
    # we don't want to lose all progress. After each fold completes, we save
    # a JSON checkpoint. On re-run, completed folds are skipped.
    checkpoint_path = 'results/ch1/mcpt_checkpoint.json'
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        completed_folds      = ckpt['completed_folds']
        per_fold_null_briers = ckpt['per_fold_null_briers']
        real_briers_mcpt     = ckpt['real_briers_mcpt']
        constant_briers_mcpt = ckpt['constant_briers_mcpt']
        print(f"Resumed from checkpoint. Completed folds: {completed_folds}")
    else:
        completed_folds      = []
        per_fold_null_briers = []       # List of lists: null Brier scores per fold
        real_briers_mcpt     = []       # Real model Brier per fold
        constant_briers_mcpt = []       # Constant baseline Brier per fold

    mcpt_start_time = time.time()

    for fold_idx in HELD_OUT_FOLDS:
        # Skip folds already completed (checkpoint resume)
        if fold_idx in completed_folds:
            print(f"\nFold {fold_idx}: SKIPPED (already in checkpoint)")
            continue

        split    = splits[fold_idx]
        df_train = df.iloc[split['train_start']:split['train_end']]
        df_val   = df.iloc[split['val_start']:split['val_end']]

        X_train, y_train = create_sequences(df_train, minimal_features, best_config['lookback'])
        X_val,   y_val   = create_sequences(df_val,   minimal_features, best_config['lookback'])

        if len(X_train) < 50 or len(X_val) < 5:
            print(f"Fold {fold_idx}: SKIP (too few samples)")
            continue

        input_shape = (best_config['lookback'], len(minimal_features))

        # --- Train the REAL model (with true labels) ---
        # This gives us the "observed test statistic" for the permutation test.
        real_model = build_gru_model(input_shape, best_config)
        real_early = EarlyStopping(monitor='val_loss', patience=5,
                                   restore_best_weights=True, verbose=0)
        real_model.fit(X_train, y_train,
                       validation_split=0.2,
                       epochs=best_config['max_epochs'],
                       batch_size=best_config['batch_size'],
                       callbacks=[real_early],
                       verbose=0)
        real_pred   = real_model.predict(X_val, verbose=0).flatten()
        real_brier  = float(np.mean((real_pred - y_val) ** 2))

        # Constant baseline: predict training event rate for all samples
        const_brier = float(np.mean((y_train.mean() - y_val) ** 2))

        real_briers_mcpt.append(real_brier)
        constant_briers_mcpt.append(const_brier)

        print(f"\nFold {fold_idx} | Real GRU Brier: {real_brier:.6f} | "
              f"Constant: {const_brier:.6f}")

        del real_model
        tf.keras.backend.clear_session()
        gc.collect()

        # --- PERMUTATION LOOP ---
        # For each permutation:
        # 1. Shuffle ONLY the training labels (features stay in original order)
        # 2. Train the same architecture with same hyperparameters
        # 3. Evaluate on the REAL (unshuffled) validation set
        # This destroys the feature-label relationship while preserving:
        # - Feature distribution and temporal structure
        # - Label distribution (same number of 0s and 1s)
        # - Model capacity and training procedure
        fold_null_briers = []
        fold_start       = time.time()

        for perm_idx in range(MCPT_PERMUTATIONS_PER_FOLD):

            # Shuffle training labels -- this is the KEY step.
            # np.random.permutation returns a NEW array (doesn't modify in-place).
            # After shuffling, features have NO predictive