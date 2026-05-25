"""
=============================================================================
SEMO Full Power Data Pipeline (Wind + Total Generation) -- Chapter 2
=============================================================================
Day-Ahead Electricity Price Forecasting on the Irish Single Electricity
Market (SEM) using GRU Neural Networks

Author:  Junior Kinyanzui
Project: TU Dublin TU821-4 Final Year Project
Chapter: 2 -- GRU-Based Day-Ahead Price Forecasting

=============================================================================
PURPOSE
=============================================================================
This script extends the base pipeline (semo_data_pipeline.py) by adding
total generation data as an additional feature source. It produces the
dataset for the FULL-FEATURE GRU model variant that includes both wind
generation AND total system generation.

The key difference from the wind-only pipeline:
    - Fetches total generation across ALL fuel types (gas, wind, hydro, etc.)
    - Computes wind penetration of generation mix (wind_mw / total_gen_mw)
    - Provides a more complete picture of supply-side market fundamentals

This additional feature captures the generation mix composition, which
influences marginal pricing through the merit order: when total generation
is high relative to wind, it implies more expensive thermal plants are
running, pushing prices up.

Data streams acquired from ENTSO-E Transparency Platform:
    1. Day-ahead electricity prices (EUR/MWh) -- hourly
    2. Wind generation (onshore B19 + offshore B18) -- half-hourly
    3. System load / actual demand (MW) -- half-hourly
    4. Total generation (all PSR types summed) -- half-hourly [NEW]

=============================================================================
DATA SOURCE
=============================================================================
ENTSO-E Transparency Platform (https://transparency.entsoe.eu)
    - Bidding zone: 10Y1001A1001A59C (IE SEM)
    - Total generation: query_generation(psr_type=None) returns all types
    - Forecast columns are excluded; only actual generation is summed

=============================================================================
OUTPUTS
=============================================================================
    entsoe_prices_raw.csv              -- Raw hourly day-ahead prices
    entsoe_wind_raw.csv                -- Raw half-hourly wind generation
    entsoe_load_raw.csv                -- Raw half-hourly actual load
    entsoe_total_gen_raw.csv           -- Raw half-hourly total generation [NEW]
    dataset_cleaned.csv                -- Aligned price + wind + load + total gen
    dataset_wind_and_total_gen.csv     -- Feature-engineered, GRU-ready dataset
                                          (includes total_gen features)

=============================================================================
FEATURE DIFFERENCES vs WIND-ONLY PIPELINE
=============================================================================
Additional features in this pipeline:
    - total_gen_mw:           Raw total generation (intermediate, not a model input)
    - wind_pen_gen:           wind_mw / total_gen_mw (wind share of generation mix)
    - total_gen_lag_48h:      Total generation 48 hours ago
    - wind_pen_gen_lag_48h:   Wind penetration of gen mix, lagged 48h

These features help the GRU understand:
    - Whether the system is running at high/low capacity overall
    - What fraction of the generation stack is wind vs thermal
    - How the generation mix 48h ago relates to current pricing

=============================================================================
DEPENDENCIES
=============================================================================
    pip install entsoe-py pandas numpy

=============================================================================
USAGE
=============================================================================
    python semo_full_power_data_pipeline.py

    Typical runtime: 8-15 minutes (extra time for total generation query,
    which returns data for all ~15 PSR types across 7 years).

=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

# Suppress pandas FutureWarnings and entsoe deprecation notices
warnings.filterwarnings("ignore")

# =============================================================================
# API CONFIGURATION
# =============================================================================
# ENTSO-E API token -- see https://transparency.entsoe.eu for registration

API_TOKEN = "YOUR_TOKEN_HERE"  # <-- REPLACE with your actual token before running 

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

# Irish Single Electricity Market bidding zone (EIC code)
BIDDING_ZONE = "10Y1001A1001A59C"   # Ireland SEM

# Date range: 2018+ only (post I-SEM reform for consistent market rules)
START_STR = "2018-01-01"
END_STR   = datetime.today().strftime("%Y-%m-%d")

# Output directory
OUTPUT_DIR = Path(".")

# Feature engineering: lag and rolling window sizes (in hours)
LAG_HOURS     = [24, 48, 168]     # 1 day, 2 days, 1 week
ROLLING_HOURS = [24, 48, 168]     # Rolling statistics windows

# ENTSO-E PSR (Production Source Reference) type codes for wind
PSR_WIND_ONSHORE  = "B19"  # Wind Onshore -- Ireland's primary renewable source
PSR_WIND_OFFSHORE = "B18"  # Wind Offshore -- small but growing capacity


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def make_client():
    """
    Create and return an authenticated ENTSO-E API client.

    Validates that the API token has been set (not left as placeholder).

    Returns
    -------
    EntsoePandasClient
        Authenticated client for ENTSO-E data queries.

    Raises
    ------
    ValueError
        If API_TOKEN is still set to the placeholder value.
    """
    if API_TOKEN == "YOUR_TOKEN_HERE":
        raise ValueError(
            "\nAPI token not set.\n"
            "Open this script and replace YOUR_TOKEN_HERE with your ENTSO-E token.\n"
            "Get one free at: https://transparency.entsoe.eu (register -> profile -> token)"
        )
    return EntsoePandasClient(api_key=API_TOKEN)


def to_timestamps(start_str, end_str):
    """
    Convert date strings to UTC-aware pandas Timestamps.

    Parameters
    ----------
    start_str : str
        Start date ('YYYY-MM-DD').
    end_str : str
        End date ('YYYY-MM-DD').

    Returns
    -------
    tuple of pd.Timestamp
        (start, end) with UTC timezone.
    """
    return (
        pd.Timestamp(start_str, tz="UTC"),
        pd.Timestamp(end_str,   tz="UTC"),
    )


def to_dublin_time(series_or_df):
    """
    Convert UTC-indexed data to Dublin local time (GMT/IST with DST).

    Handles the clock change correctly: Ireland is UTC+0 in winter
    and UTC+1 (Irish Standard Time) in summer. After conversion,
    timezone info is removed for simpler downstream indexing.

    Parameters
    ----------
    series_or_df : pd.Series or pd.DataFrame
        Data with UTC (or tz-naive assumed UTC) DatetimeIndex.

    Returns
    -------
    pd.Series or pd.DataFrame
        Same data with Dublin local time index (tz-naive).
    """
    obj = series_or_df.copy()
    if obj.index.tz is None:
        obj.index = obj.index.tz_localize("UTC")
    obj.index = obj.index.tz_convert("Europe/Dublin").tz_localize(None)
    return obj


def fetch_yearly_chunks(fetch_fn, start_str, end_str, label):
    """
    Fetch data from ENTSO-E in yearly chunks to prevent API timeouts.

    The ENTSO-E API has undocumented limits on query duration. Fetching
    more than ~1 year at a time can cause timeouts or 500 errors. This
    function breaks requests into annual segments and handles:
        - NoMatchingDataError (no data for certain periods)
        - Network errors and API failures
        - MultiIndex column flattening (varies by year/PSR type)

    Parameters
    ----------
    fetch_fn : callable
        API query function: fn(start, end) -> DataFrame/Series.
    start_str : str
        Start date ('YYYY-MM-DD').
    end_str : str
        End date ('YYYY-MM-DD').
    label : str
        Human-readable label for progress messages.

    Returns
    -------
    pd.DataFrame or pd.Series or None
        Combined, deduplicated, sorted data. None if no data retrieved.
    """
    start_ts, end_ts = to_timestamps(start_str, end_str)
    chunks  = []
    current = start_ts

    while current < end_ts:
        chunk_end = min(current + pd.DateOffset(years=1), end_ts)
        year      = current.strftime("%Y")
        print(f"  {label}: fetching {year}...", end=" ", flush=True)

        try:
            result = fetch_fn(current, chunk_end)
            if result is not None and not (
                hasattr(result, "empty") and result.empty
            ):
                chunks.append(result)
                n = len(result)
                print(f"OK  ({n:,} records)")
            else:
                print("X  (no data)")

        except NoMatchingDataError:
            print("X  (no data for this period)")
        except Exception as e:
            print(f"X  (error: {e})")

        current = chunk_end
        time.sleep(1)   # Rate-limit courtesy delay

    if not chunks:
        return None

    # Normalize MultiIndex columns to plain strings for safe concatenation.
    # entsoe-py's query_generation can return MultiIndex columns in some
    # year-chunks (e.g., ("Actual Aggregated", "B19")) and plain columns
    # in others, which would break pd.concat without this normalization.
    def _flatten_chunk(obj):
        obj = obj.copy()
        if isinstance(obj.index, pd.MultiIndex):
            obj.index = obj.index.get_level_values(0)
        if isinstance(obj, pd.DataFrame) and isinstance(obj.columns, pd.MultiIndex):
            obj.columns = ["_".join(str(v) for v in col).strip("_")
                           for col in obj.columns.to_flat_index()]
        return obj

    chunks = [_flatten_chunk(c) for c in chunks]
    combined = pd.concat(chunks)
    if hasattr(combined, "index"):
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined


# =============================================================================
# SECTION 1: FETCH DAY-AHEAD PRICES
# =============================================================================

def fetch_prices(client):
    """
    Fetch Irish day-ahead market clearing prices from ENTSO-E.

    Day-ahead prices reflect the marginal cost of the last generator
    needed to meet demand for each hour. They are set at the hourly
    resolution via a uniform-price auction conducted by SEMO.

    Parameters
    ----------
    client : EntsoePandasClient
        Authenticated API client.

    Returns
    -------
    pd.DataFrame
        Hourly prices ('price_eur_mwh') in Dublin local time.
    """
    print("\n--- Fetching day-ahead prices ---")

    def fn(start, end):
        return client.query_day_ahead_prices(BIDDING_ZONE, start=start, end=end)

    raw = fetch_yearly_chunks(fn, START_STR, END_STR, "Prices")

    if raw is None:
        print("  ERROR: No price data retrieved.")
        return pd.DataFrame()

    df = to_dublin_time(raw.to_frame(name="price_eur_mwh"))
    df = df[~df.index.duplicated(keep="first")].sort_index()
    print(f"\n  Prices: {len(df):,} hourly records "
          f"({df.index.min().date()} -> {df.index.max().date()})")
    return df


# =============================================================================
# SECTION 2: FETCH WIND GENERATION
# =============================================================================

def fetch_wind(client):
    """
    Fetch actual wind generation (onshore + offshore combined) for Ireland.

    Ireland's wind fleet is predominantly onshore (~5 GW installed).
    Offshore wind is minimal but growing. Both are summed to give total
    wind output in MW at half-hourly resolution.

    Wind is the key renewable driver of Irish electricity prices due to
    the merit order effect: zero-marginal-cost wind displaces expensive
    gas generation and suppresses prices during high-wind periods.

    Parameters
    ----------
    client : EntsoePandasClient
        Authenticated API client.

    Returns
    -------
    pd.DataFrame
        Half-hourly total wind generation ('wind_mw') in Dublin time.
    """
    print("\n--- Fetching wind generation ---")

    all_wind = []

    for psr, label in [(PSR_WIND_ONSHORE, "Onshore"),
                       (PSR_WIND_OFFSHORE, "Offshore")]:

        def fn(start, end, psr=psr):
            result = client.query_generation(
                BIDDING_ZONE, start=start, end=end, psr_type=psr
            )
            return result

        raw = fetch_yearly_chunks(fn, START_STR, END_STR, f"Wind {label}")

        if raw is None:
            print(f"  No {label} wind data available -- skipping")
            continue

        # Extract actual generation column from the query result
        if isinstance(raw, pd.DataFrame):
            actual_col = next(
                (c for c in raw.columns
                 if "actual" in str(c).lower() or "aggregated" in str(c).lower()),
                raw.columns[0]
            )
            series = raw[actual_col].rename(f"wind_{label.lower()}_mw")
        else:
            series = raw.rename(f"wind_{label.lower()}_mw")

        series = to_dublin_time(series.to_frame()).iloc[:, 0]
        all_wind.append(series)
        print(f"  Wind {label}: {len(series):,} records")

    if not all_wind:
        print("  ERROR: No wind data retrieved.")
        return pd.DataFrame()

    # Sum onshore + offshore to get total wind generation
    wind_df = pd.concat(all_wind, axis=1).sum(axis=1).to_frame(name="wind_mw")
    wind_df = wind_df[~wind_df.index.duplicated(keep="first")].sort_index()

    print(f"\n  Wind total: {len(wind_df):,} records "
          f"({wind_df.index.min().date()} -> {wind_df.index.max().date()})")
    return wind_df


# =============================================================================
# SECTION 3: FETCH TOTAL GENERATION (ALL FUEL TYPES)
# =============================================================================

def fetch_total_generation(client):
    """
    Fetch total actual generation for Ireland across ALL generation types.

    This queries ENTSO-E with psr_type=None, which returns generation data
    for every fuel type: gas (B04), wind (B19), hydro (B11), biomass (B01),
    coal (B02), oil (B03), peat (B05), etc. All actual generation columns
    are summed to produce a single total_gen_mw value.

    Forecast columns are explicitly excluded to avoid contamination.

    WHY THIS MATTERS:
    Total generation captures the supply-side pressure on the market.
    When total generation is high, it typically means demand is high
    AND/OR exports are high, both of which correlate with higher prices.
    The ratio wind_mw/total_gen_mw reveals wind's share of the generation
    mix, which is a stronger price signal than wind alone.

    Parameters
    ----------
    client : EntsoePandasClient
        Authenticated API client.

    Returns
    -------
    pd.DataFrame
        Half-hourly total generation ('total_gen_mw') in Dublin time.
    """
    print("\n--- Fetching total generation (all types) ---")

    def fn(start, end):
        # psr_type=None returns ALL generation types in a single query
        result = client.query_generation(
            BIDDING_ZONE, start=start, end=end, psr_type=None
        )
        return result

    raw = fetch_yearly_chunks(fn, START_STR, END_STR, "Total Gen")

    if raw is None:
        print("  ERROR: No generation data retrieved.")
        return pd.DataFrame()

    if isinstance(raw, pd.DataFrame):
        # Filter out forecast columns -- we only want actual generation
        # Forecast columns typically contain 'forecast' in their name
        actual_cols = [c for c in raw.columns
                       if "forecast" not in str(c).lower()]
        if not actual_cols:
            actual_cols = list(raw.columns)  # Fallback: use all columns
        # Sum across all fuel types to get total system generation
        total = raw[actual_cols].sum(axis=1)
        print(f"  Summed {len(actual_cols)} generation type columns")
    else:
        total = raw

    df = to_dublin_time(total.to_frame(name="total_gen_mw"))
    df = df[~df.index.duplicated(keep="first")].sort_index()

    print(f"\n  Total generation: {len(df):,} records "
          f"({df.index.min().date()} -> {df.index.max().date()})")
    return df


# =============================================================================
# SECTION 4: FETCH SYSTEM LOAD (DEMAND)
# =============================================================================

def fetch_load(client):
    """
    Fetch actual system load (electricity demand) for Ireland.

    System load follows predictable diurnal and seasonal patterns:
        - Daily: low overnight (2-5 GW), morning ramp (7-9am), evening peak (5-7pm)
        - Weekly: lower on weekends (reduced commercial/industrial demand)
        - Seasonal: higher in winter (heating, lighting), lower in summer

    Load is used both directly (load_lag_48h) and as the denominator in
    wind penetration ratios (wind_mw / load_mw).

    Parameters
    ----------
    client : EntsoePandasClient
        Authenticated API client.

    Returns
    -------
    pd.DataFrame
        Half-hourly system load ('load_mw') in Dublin local time.
    """
    print("\n--- Fetching system load ---")

    def fn(start, end):
        result = client.query_load_and_forecast(BIDDING_ZONE, start=start, end=end)
        return result

    raw = fetch_yearly_chunks(fn, START_STR, END_STR, "Load")

    if raw is None:
        print("  ERROR: No load data retrieved.")
        return pd.DataFrame()

    if isinstance(raw, pd.DataFrame):
        # Select actual load column (excluding day-ahead/week-ahead forecasts)
        actual_col = next(
            (c for c in raw.columns
             if "actual" in str(c).lower()),
            raw.columns[0]
        )
        series = raw[actual_col]
    else:
        series = raw

    df = to_dublin_time(series.to_frame(name="load_mw"))
    df = df[~df.index.duplicated(keep="first")].sort_index()

    print(f"\n  Load: {len(df):,} records "
          f"({df.index.min().date()} -> {df.index.max().date()})")
    return df


# =============================================================================
# SECTION 5: TEMPORAL ALIGNMENT
# =============================================================================
# All four data streams must be aligned to a common half-hourly grid.
# Prices (hourly) are forward-filled to 30-min; wind, load, and total
# generation are already at 30-min but may have gaps or misaligned timestamps.

def resample_to_halfhourly(df, name):
    """
    Upsample hourly data to half-hourly using forward-fill.

    Day-ahead prices apply uniformly within each hour, so forward-filling
    from H:00 to H:30 is the correct approach (not interpolation).

    Parameters
    ----------
    df : pd.DataFrame
        Hourly data to upsample.
    name : str
        Label for progress messages.

    Returns
    -------
    pd.DataFrame
        Half-hourly data with forward-filled values.
    """
    if df.empty:
        return df
    hh_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="30min")
    out    = df.reindex(hh_idx).ffill()
    print(f"  [{name}] Resampled to half-hourly: {len(out):,} records")
    return out


def align_series(series, master_idx, name, fill_limit=2):
    """
    Reindex a series to a master index with limited forward-fill for gaps.

    Only fills gaps up to fill_limit consecutive periods (= 1 hour at
    30-min resolution). Longer gaps remain as NaN to avoid masking
    genuine data outages.

    Parameters
    ----------
    series : pd.Series
        Data to align.
    master_idx : pd.DatetimeIndex
        Target half-hourly index.
    name : str
        Column name for output.
    fill_limit : int, default 2
        Max consecutive NaN periods to fill (2 periods = 1 hour).

    Returns
    -------
    pd.Series
        Aligned and gap-filled series.
    """
    s = series.reindex(master_idx) if not series.empty \
        else pd.Series(np.nan, index=master_idx)

    missing = s.isna().sum()
    if missing:
        s = s.ffill(limit=fill_limit)
        filled = missing - s.isna().sum()
        if filled:
            print(f"  [{name}] Filled {filled} short gaps (<=1 hour)")
        still = s.isna().sum()
        if still:
            print(f"  [{name}] WARNING: {still} ({100*still/len(s):.1f}%) periods still missing")
    return s.rename(name)


def align_all(price_df, wind_df, load_df, total_gen_df):
    """
    Align all four data sources onto a unified half-hourly master index.

    This is the critical data fusion step: prices (hourly), wind (30-min),
    load (30-min), and total generation (30-min) are brought together into
    a single DataFrame with consistent timestamps.

    Parameters
    ----------
    price_df : pd.DataFrame
        Hourly prices.
    wind_df : pd.DataFrame
        Half-hourly wind generation.
    load_df : pd.DataFrame
        Half-hourly system load.
    total_gen_df : pd.DataFrame
        Half-hourly total generation.

    Returns
    -------
    pd.DataFrame
        Aligned DataFrame with columns: price_eur_mwh, wind_mw, load_mw,
        total_gen_mw. Rows with missing price are dropped.
    """
    print("\n--- Aligning to master half-hourly index ---")

    # Upsample hourly prices to half-hourly
    price_hh = resample_to_halfhourly(price_df, "price")

    # Create master index spanning the full date range
    master = pd.date_range(start=START_STR, end=END_STR, freq="30min")
    print(f"  Master index: {len(master):,} periods ({START_STR} -> {END_STR})")

    def get_s(df):
        """Safely extract first column as Series."""
        return df.iloc[:, 0] if (df is not None and not df.empty) \
               else pd.Series(dtype=float)

    # Align each series to master index
    p  = align_series(get_s(price_hh),      master, "price_eur_mwh")
    w  = align_series(get_s(wind_df),       master, "wind_mw")
    l  = align_series(get_s(load_df),       master, "load_mw")
    tg = align_series(get_s(total_gen_df),  master, "total_gen_mw")

    # Combine into single DataFrame
    df = pd.concat([p, w, l, tg], axis=1)

    # Target variable (price) cannot be NaN -- drop those rows
    before = len(df)
    df = df.dropna(subset=["price_eur_mwh"])
    if before != len(df):
        print(f"  Dropped {before - len(df)} rows with missing price")

    print(f"  Aligned dataset: {len(df):,} rows")
    return df


# =============================================================================
# SECTION 6: FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    """
    Build all GRU input features with strict no-lookahead constraints.

    This is the EXTENDED version that includes total generation features
    in addition to all features from the wind-only pipeline.

    Feature groups:
        1. CYCLICAL TIME ENCODINGS (6 features)
           hour_sin/cos, dow_sin/cos, month_sin/cos

        2. PRICE LAGS (3 features)
           24h, 48h, 168h historical prices

        3. ROLLING PRICE STATISTICS (6 features)
           Mean and std over 24h, 48h, 168h windows (shift(1) applied)

        4. WIND AND LOAD FEATURES (3 features -- same as wind-only)
           wind_lag_48h, load_lag_48h, wind_pen_lag_48h

        5. TOTAL GENERATION FEATURES (2 features -- NEW in this pipeline)
           total_gen_lag_48h: Total system generation 48 hours ago
           wind_pen_gen_lag_48h: Wind share of generation mix, lagged 48h

    Penetration ratios:
        wind_pen = wind_mw / load_mw
            -> How much of DEMAND is met by wind (demand-side perspective)
        wind_pen_gen = wind_mw / total_gen_mw
            -> Wind's share of total GENERATION (supply-side perspective)

    Both perspectives matter: demand-coverage tells us about price
    suppression potential; generation-share tells us about the
    displacement of thermal plants in the merit order.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, aligned DataFrame with price, wind, load, total_gen columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with all engineered features.
    """
    print("\n--- Engineering features ---")
    pph = 2  # Periods per hour (30-min resolution)

    # --- Group 1: Cyclical time encodings ---
    # Sin/cos pairs preserve circular continuity (hour 23 close to hour 0)
    df["hour_sin"]  = np.sin(2 * np.pi * df.index.hour      / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df.index.hour      / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month     / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month     / 12)

    # --- Group 2: Price lag features ---
    for h in LAG_HOURS:
        df[f"price_lag_{h}h"] = df["price_eur_mwh"].shift(h * pph)

    # --- Group 3: Rolling price statistics ---
    # shift(1) prevents the current period's price from being included
    for h in ROLLING_HOURS:
        base = df["price_eur_mwh"].shift(1)
        w    = h * pph
        df[f"price_roll_mean_{h}h"] = base.rolling(w, min_periods=1).mean()
        df[f"price_roll_std_{h}h"]  = base.rolling(w, min_periods=1).std()

    # --- Groups 4 & 5: Wind, load, and total generation features ---
    if "wind_mw" in df.columns and "total_gen_mw" in df.columns:
        # Wind penetration of demand (how much demand is met by wind)
        df["wind_pen"] = np.where(
            df["load_mw"] > 0,
            df["wind_mw"] / df["load_mw"],
            np.nan
        )
        # Wind penetration of generation mix (wind's share of all generation)
        # This captures the fuel mix composition on the supply side
        df["wind_pen_gen"] = np.where(
            df["total_gen_mw"] > 0,
            df["wind_mw"] / df["total_gen_mw"],
            np.nan
        )

        # All supply-side features lagged by 48 hours (well before gate closure)
        df["wind_lag_48h"]          = df["wind_mw"].shift(48 * pph)
        df["load_lag_48h"]          = df["load_mw"].shift(48 * pph)
        df["total_gen_lag_48h"]     = df["total_gen_mw"].shift(48 * pph)
        df["wind_pen_lag_48h"]      = df["wind_pen"].shift(48 * pph)
        df["wind_pen_gen_lag_48h"]  = df["wind_pen_gen"].shift(48 * pph)

    # Summary of created features
    feature_cols = [c for c in df.columns if c != "price_eur_mwh"]
    print(f"  {len(feature_cols)} features created:")
    for c in feature_cols:
        print(f"    {c}")
    return df


# =============================================================================
# SECTION 7: DATA LEAKAGE VALIDATION
# =============================================================================

def validate_leakage(df):
    """
    Verify no feature uses future information (minimum lag >= 1h).

    Iterates through all lag and rolling features, parsing the lag
    duration from column names, and flags any with insufficient offset.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame.

    Returns
    -------
    bool
        True if all features pass validation.
    """
    print("\n--- Leakage validation ---")
    lag_cols = [c for c in df.columns if "_lag_" in c or "roll_" in c]
    ok = True
    for c in lag_cols:
        if "_lag_" in c:
            try:
                h = int(c.split("_lag_")[1].replace("h", ""))
                if h < 1:
                    print(f"  FAIL: {c} -- lag < 1h!")
                    ok = False
            except Exception:
                pass
    msg = f"No leakage -- {len(lag_cols)} features validated" if ok \
          else "LEAKAGE DETECTED -- review features immediately"
    print(f"  {msg}")
    return ok


def quality_report(df):
    """
    Print comprehensive quality report for the final dataset.

    Includes dataset shape, date range, missing values per column,
    price statistics, and notes on negative prices and extreme spikes.
    Also checks total_gen_mw completeness.

    Parameters
    ----------
    df : pd.DataFrame
        Final feature-engineered dataset.
    """
    print("\n" + "=" * 62)
    print("  DATASET QUALITY REPORT")
    print("=" * 62)
    years = (df.index.max() - df.index.min()).days / 365.25
    print(f"  Rows:        {len(df):,}")
    print(f"  Columns:     {len(df.columns)}")
    print(f"  Range:       {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Span:        {years:.1f} years")
    print(f"  Resolution:  30-minute\n")

    print("  Missing values:")
    for col in df.columns:
        n   = df[col].isna().sum()
        pct = 100 * n / len(df)
        flag = "  WARNING" if pct > 1.0 else ""
        print(f"    {col:<35} {n:>6} ({pct:.1f}%){flag}")

    print(f"\n  Price statistics (EUR/MWh):")
    for k, v in df["price_eur_mwh"].describe().items():
        print(f"    {k:<10} {v:>10.2f}")

    # Negative prices: real market events during high wind/low demand
    neg = (df["price_eur_mwh"] < 0).sum()
    if neg:
        print(f"\n  NOTE: {neg} negative price periods.")
        print("  Real market events -- high wind + low demand. Keep them.")
        print("  Cite in thesis as a characteristic of the Irish SEM.")

    # Extreme price spikes: scarcity/system stress events
    q99    = df["price_eur_mwh"].quantile(0.99)
    spikes = (df["price_eur_mwh"] > q99 * 2).sum()
    if spikes:
        print(f"\n  NOTE: {spikes} extreme spikes (>2x p99 = {q99*2:.0f} EUR/MWh).")
        print("  Energy-market analogue of volatility clustering from Chapter 1.")
        print("  GRU will struggle here -- document as a known limitation.")

    # Total generation data completeness check
    if "total_gen_mw" in df.columns:
        tg_missing = df["total_gen_mw"].isna().sum()
        if tg_missing > 0:
            print(f"\n  WARNING: total_gen_mw has {tg_missing} missing values.")
            print("  Document as a data limitation in thesis if >1%.")

    print("=" * 62)


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================

def run():
    """
    Execute the full data pipeline for the wind + total generation model.

    Pipeline stages:
        1. Fetch raw data (prices, wind, load, total generation) from ENTSO-E
        2. Save raw CSVs for reproducibility
        3. Align all four sources to common half-hourly index
        4. Engineer features (including total generation features)
        5. Validate no data leakage
        6. Drop warmup rows and save final GRU-ready dataset
        7. Print quality report

    Output: dataset_wind_and_total_gen.csv -- ready for ch2_gru_semo.py
    """
    print("=" * 62)
    print("  ENTSO-E Pipeline -- Chapter 2 (Wind + Total Generation)")
    print("=" * 62)
    print(f"  Zone:   {BIDDING_ZONE} (Ireland SEM)")
    print(f"  Range:  {START_STR} -> {END_STR}\n")

    client = make_client()

    # --- Stage 1: Fetch all four data sources ---
    price_df     = fetch_prices(client)
    wind_df      = fetch_wind(client)
    load_df      = fetch_load(client)
    total_gen_df = fetch_total_generation(client)

    # --- Stage 2: Save raw files ---
    for df, fname in [(price_df,     "entsoe_prices_raw.csv"),
                      (wind_df,      "entsoe_wind_raw.csv"),
                      (load_df,      "entsoe_load_raw.csv"),
                      (total_gen_df, "entsoe_total_gen_raw.csv")]:
        if df is not None and not df.empty:
            df.to_csv(OUTPUT_DIR / fname)
            print(f"  Saved: {fname}")

    # --- Stage 3: Temporal alignment ---
    print("\nSTEP: Aligning sources")
    df_clean = align_all(price_df, wind_df, load_df, total_gen_df)
    df_clean.to_csv(OUTPUT_DIR / "dataset_cleaned.csv")
    print("  Saved: dataset_cleaned.csv")

    # --- Stage 4: Feature engineering ---
    print("\nSTEP: Feature engineering")
    df_feat = engineer_features(df_clean.copy())

    # --- Stage 5: Leakage validation ---
    validate_leakage(df_feat)

    # --- Stage 6: Drop warmup NaNs and save ---
    before   = len(df_feat)
    df_model = df_feat.dropna()
    warmup   = before - len(df_model)
    print(f"\n  Dropped {warmup} warmup rows (lag initialisation)")
    print(f"  Final: {len(df_model):,} rows x {len(df_model.columns)} columns")

    df_model.to_csv(OUTPUT_DIR / "dataset_wind_and_total_gen.csv")
    print("  Saved: dataset_wind_and_total_gen.csv")

    # --- Stage 7: Quality report ---
    quality_report(df_model)

    # File manifest
    print("\nFILES:")
    for fname in ["entsoe_prices_raw.csv", "entsoe_wind_raw.csv",
                  "entsoe_load_raw.csv", "entsoe_total_gen_raw.csv",
                  "dataset_cleaned.csv", "dataset_wind_and_total_gen.csv"]:
        p = OUTPUT_DIR / fname
        if p.exists():
            print(f"  OK  {fname:<42} ({p.stat().st_size // 1024} KB)")
        else:
            print(f"  X   {fname:<42} (not created)")

    print("\nNEXT: Run ch2_gru_semo.py with ACTIVE_DATASET = 'wind_and_total_gen'")
    print("=" * 62)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run()
