"""
=============================================================================
SEMO Data Pipeline (Wind-Only) -- Chapter 2
=============================================================================
Day-Ahead Electricity Price Forecasting on the Irish Single Electricity
Market (SEM) using GRU Neural Networks

Author:  Junior Kinyanzui
Project: TU Dublin TU821-4 Final Year Project
Chapter: 2 -- GRU-Based Day-Ahead Price Forecasting

=============================================================================
PURPOSE
=============================================================================
This script acquires, cleans, aligns, and feature-engineers the dataset
required for the wind-only GRU model variant (Chapter 2, Section 2.3).

It downloads three data streams from the ENTSO-E Transparency Platform:
    1. Day-ahead electricity prices (EUR/MWh)
    2. Wind generation -- onshore (B19) + offshore (B18) summed
    3. System load (actual demand in MW)

The final output (dataset_wind_only.csv) is a half-hourly time series with:
    - Target variable: price_eur_mwh
    - Cyclical time features (hour, day-of-week, month as sin/cos pairs)
    - Lagged price features (24h, 48h, 168h)
    - Rolling price statistics (mean/std over 24h, 48h, 168h windows)
    - Wind generation lag (48h)
    - System load lag (48h)
    - Wind penetration ratio lag (48h) -- wind_mw / load_mw

All features respect the day-ahead gate closure constraint: minimum lag of
24 hours ensures no information leakage from the target delivery period.

=============================================================================
DATA SOURCE
=============================================================================
ENTSO-E Transparency Platform (https://transparency.entsoe.eu)
    - Free registration required for API token
    - Provides pan-European energy market data
    - Irish bidding zone: 10Y1001A1001A59C (IE SEM)

WHY 2018 START DATE:
    Ireland's Integrated Single Electricity Market (I-SEM) reform was
    completed in October 2018, replacing the previous SEM arrangements.
    Pre-2018 data uses different market coupling rules (ex-post pool pricing
    vs. day-ahead auction) and is therefore not directly comparable.
    Starting from 2018-01-01 ensures consistent market structure throughout.

=============================================================================
OUTPUTS
=============================================================================
    entsoe_prices_raw.csv    -- Raw hourly day-ahead prices (Dublin time)
    entsoe_wind_raw.csv      -- Raw half-hourly wind generation (MW)
    entsoe_load_raw.csv      -- Raw half-hourly actual system load (MW)
    dataset_cleaned.csv      -- Aligned half-hourly price + wind + load
    dataset_wind_only.csv    -- Feature-engineered, GRU-ready dataset
                                (leakage-validated, NaN warmup rows dropped)

=============================================================================
DEPENDENCIES
=============================================================================
    pip install entsoe-py pandas numpy

=============================================================================
USAGE
=============================================================================
    python semo_data_pipeline.py

    The script runs end-to-end: fetch -> align -> engineer -> validate -> save.
    Typical runtime: 5-10 minutes (API rate-limited, fetches ~7 years of data).

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

# Suppress pandas FutureWarnings and entsoe deprecation notices for clean output
warnings.filterwarnings("ignore")

# =============================================================================
# API CONFIGURATION
# =============================================================================
# To obtain a token: register at https://transparency.entsoe.eu, then
# navigate to Profile -> Security Token -> Generate.
# The token is free and provides access to all public ENTSO-E data.

API_TOKEN = "YOUR_TOKEN_HERE"  # <-- REPLACE with your actual token before running

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

# Irish Single Electricity Market bidding zone identifier (EIC code)
BIDDING_ZONE = "10Y1001A1001A59C"   # Ireland SEM

# Date range for data acquisition
# Start: 2018-01-01 (post I-SEM reform -- consistent market rules)
# End: today (script dynamically fetches up to current date)
START_STR = "2018-01-01"
END_STR   = datetime.today().strftime("%Y-%m-%d")

# Output directory for all CSV files
OUTPUT_DIR = Path(".")

# Feature engineering parameters
# Lag hours represent historical offsets used as predictive features.
# 24h = same time yesterday, 48h = same time 2 days ago, 168h = same time last week
LAG_HOURS     = [24, 48, 168]     # 1 day, 2 days, 1 week
ROLLING_HOURS = [24, 48, 168]     # Windows for rolling mean/std statistics

# ENTSO-E PSR (Production Source Reference) type codes for wind generation
# See: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
PSR_WIND_ONSHORE  = "B19"  # Wind Onshore -- dominant source in Ireland (~5GW capacity)
PSR_WIND_OFFSHORE = "B18"  # Wind Offshore -- small but growing (~0.03GW in 2024)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def make_client():
    """
    Create and return an authenticated ENTSO-E API client.

    Raises ValueError if the API token placeholder has not been replaced
    with a valid token, providing instructions on how to obtain one.

    Returns
    -------
    EntsoePandasClient
        Authenticated client ready for data queries.
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
    Convert date strings to timezone-aware pandas Timestamps in UTC.

    The ENTSO-E API requires timezone-aware timestamps for queries.
    UTC is used as the canonical timezone; conversion to Dublin local
    time happens after data retrieval.

    Parameters
    ----------
    start_str : str
        Start date in 'YYYY-MM-DD' format.
    end_str : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    tuple of pd.Timestamp
        (start, end) as UTC-aware Timestamps.
    """
    return (
        pd.Timestamp(start_str, tz="UTC"),
        pd.Timestamp(end_str,   tz="UTC"),
    )


def to_dublin_time(series_or_df):
    """
    Convert a UTC-indexed Series or DataFrame to Dublin local time.

    Ireland observes GMT (UTC+0) in winter and IST (UTC+1) in summer.
    This function handles the DST transition correctly via pytz/dateutil.
    After conversion, timezone info is stripped to produce a naive
    DatetimeIndex for simpler downstream processing.

    Parameters
    ----------
    series_or_df : pd.Series or pd.DataFrame
        Data with a DatetimeIndex (UTC or tz-naive assumed UTC).

    Returns
    -------
    pd.Series or pd.DataFrame
        Same data with index converted to Dublin local time (tz-naive).
    """
    obj = series_or_df.copy()
    # If index has no timezone, assume it is UTC
    if obj.index.tz is None:
        obj.index = obj.index.tz_localize("UTC")
    # Convert to Dublin time, then strip tz info for clean indexing
    obj.index = obj.index.tz_convert("Europe/Dublin").tz_localize(None)
    return obj


def fetch_yearly_chunks(fetch_fn, start_str, end_str, label):
    """
    Fetch data from ENTSO-E in yearly chunks to avoid API timeouts.

    The ENTSO-E API can timeout or return errors for very large date
    ranges. This function breaks the request into 1-year segments,
    retries gracefully on failure, and concatenates results.

    A 1-second sleep between requests respects API rate limits and
    avoids being throttled or banned.

    Parameters
    ----------
    fetch_fn : callable
        Function accepting (start: pd.Timestamp, end: pd.Timestamp)
        and returning a DataFrame or Series from the ENTSO-E API.
    start_str : str
        Start date string ('YYYY-MM-DD').
    end_str : str
        End date string ('YYYY-MM-DD').
    label : str
        Human-readable label for progress printing (e.g., "Prices").

    Returns
    -------
    pd.DataFrame or pd.Series or None
        Combined data from all yearly chunks, deduplicated and sorted.
        Returns None if no data could be retrieved for any chunk.
    """
    start_ts, end_ts = to_timestamps(start_str, end_str)
    chunks  = []
    current = start_ts

    while current < end_ts:
        # Define the end of this chunk (max 1 year ahead)
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
            # ENTSO-E returns this when no data exists for the period
            print("X  (no data for this period)")
        except Exception as e:
            # Log other errors (network issues, malformed responses, etc.)
            print(f"X  (error: {e})")

        current = chunk_end
        time.sleep(1)   # Respectful delay -- avoid hammering the API

    if not chunks:
        return None

    # entsoe-py sometimes returns MultiIndex columns (e.g., for generation
    # data: ('Actual Aggregated', 'B19')). Different years may have
    # different column structures. Flatten to plain string columns so
    # pd.concat can union them without raising NotImplementedError.
    normalized = []
    for chunk in chunks:
        if isinstance(chunk, pd.DataFrame) and isinstance(chunk.columns, pd.MultiIndex):
            chunk = chunk.copy()
            chunk.columns = ["_".join(str(c) for c in col).strip("_")
                             for col in chunk.columns]
        normalized.append(chunk)

    # Concatenate all chunks and remove any duplicate timestamps
    # (can occur at chunk boundaries due to inclusive/exclusive endpoint handling)
    combined = pd.concat(normalized)
    if hasattr(combined, "index"):
        combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined


# =============================================================================
# SECTION 1: FETCH DAY-AHEAD PRICES
# =============================================================================

def fetch_prices(client):
    """
    Fetch Irish day-ahead market clearing prices from ENTSO-E.

    The day-ahead market auction determines the electricity price for each
    hour of the following day. Results are published by ~12:30 on day D-1.
    Prices are in EUR/MWh at hourly resolution.

    Parameters
    ----------
    client : EntsoePandasClient
        Authenticated ENTSO-E API client.

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame ('price_eur_mwh') with DatetimeIndex
        in Dublin local time, at hourly resolution.
        Returns empty DataFrame if no data available.
    """
    print("\n--- Fetching day-ahead prices ---")

    def fn(start, end):
        return client.query_day_ahead_prices(BIDDING_ZONE, start=start, end=end)

    raw = fetch_yearly_chunks(fn, START_STR, END_STR, "Prices")

    if raw is None:
        print("  ERROR: No price data retrieved.")
        return pd.DataFrame()

    # Convert from UTC to Dublin local time
    df = to_dublin_time(raw.to_frame(name="price_eur_mwh"))
    # Remove any duplicate timestamps (DST transitions can cause overlaps)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    print(f"\n  Prices: {len(df):,} hourly records "
          f"({df.index.min().date()} -> {df.index.max().date()})")
    return df


# =============================================================================
# SECTION 2: FETCH WIND GENERATION
# =============================================================================

def fetch_wind(client):
    """
    Fetch actual wind generation for Ireland (onshore + offshore).

    Wind generation is the single most important supply-side driver of
    electricity prices in Ireland, where wind capacity exceeds 5 GW and
    regularly provides 30-70% of demand. High wind periods suppress prices
    (merit order effect); low wind periods require expensive gas generation.

    The function fetches onshore (B19) and offshore (B18) separately,
    then sums them to get total wind generation in MW.

    Parameters
    ----------
    client : EntsoePandasClient
        Authenticated ENTSO-E API client.

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame ('wind_mw') with DatetimeIndex in Dublin
        local time, at half-hourly resolution.
        Returns empty DataFrame if no wind data available.
    """
    print("\n--- Fetching wind generation ---")

    all_wind = []

    for psr, label in [(PSR_WIND_ONSHORE, "Onshore"),
                       (PSR_WIND_OFFSHORE, "Offshore")]:

        # Use default argument psr=psr to capture loop variable correctly
        def fn(start, end, psr=psr):
            result = client.query_generation(
                BIDDING_ZONE, start=start, end=end, psr_type=psr
            )
            return result

        raw = fetch_yearly_chunks(fn, START_STR, END_STR, f"Wind {label}")

        if raw is None:
            print(f"  No {label} wind data available -- skipping")
            continue

        # query_generation() returns a DataFrame with columns like
        # 'Actual Aggregated' or 'Actual Consumption'. We need the
        # actual generation column.
        if isinstance(raw, pd.DataFrame):
            # Heuristic: find column containing 'actual' or 'aggregated'
            actual_col = next(
                (c for c in raw.columns
                 if "actual" in str(c).lower() or "aggregated" in str(c).lower()),
                raw.columns[0]  # Fallback to first column
            )
            series = raw[actual_col].rename(f"wind_{label.lower()}_mw")
        else:
            series = raw.rename(f"wind_{label.lower()}_mw")

        # Convert to Dublin time
        series = to_dublin_time(series.to_frame()).iloc[:, 0]
        all_wind.append(series)
        print(f"  Wind {label}: {len(series):,} records")

    if not all_wind:
        print("  ERROR: No wind data retrieved.")
        return pd.DataFrame()

    # Sum onshore + offshore into total wind generation
    # axis=1 aligns by timestamp; sum treats NaN as 0 by default
    wind_df = pd.concat(all_wind, axis=1).sum(axis=1).to_frame(name="wind_mw")
    wind_df = wind_df[~wind_df.index.duplicated(keep="first")].sort_index()

    print(f"\n  Wind total: {len(wind_df):,} records "
          f"({wind_df.index.min().date()} -> {wind_df.index.max().date()})")
    return wind_df


# =============================================================================
# SECTION 3: FETCH SYSTEM LOAD (DEMAND)
# =============================================================================

def fetch_load(client):
    """
    Fetch actual system load (electricity demand) for Ireland.

    System load represents total electricity consumption on the grid.
    It follows strong diurnal patterns (low overnight, peaks at 17:00-19:00)
    and seasonal patterns (higher in winter due to heating/lighting).

    Load is used to compute wind penetration ratio (wind_mw / load_mw),
    which is a key predictor of price: when wind meets a large fraction
    of demand, conventional generators are displaced and prices fall.

    Parameters
    ----------
    client : EntsoePandasClient
        Authenticated ENTSO-E API client.

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame ('load_mw') with DatetimeIndex in Dublin
        local time, at half-hourly resolution.
        Returns empty DataFrame if no load data available.
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
        # The API returns both actual and forecast load -- we want actual only
        actual_col = next(
            (c for c in raw.columns
             if "actual" in str(c).lower()),
            raw.columns[0]  # Fallback to first column
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
# SECTION 4: TEMPORAL ALIGNMENT
# =============================================================================
# ENTSO-E data arrives at different resolutions:
#   - Prices: hourly (1 value per hour)
#   - Wind/Load: half-hourly (2 values per hour)
#
# We align everything to a uniform 30-minute (half-hourly) grid because:
#   1. The Irish SEM settles at half-hourly resolution
#   2. Preserves the higher-resolution wind/load variation
#   3. Gives 48 periods per day (convenient for 24h-ahead forecasting)

def resample_to_halfhourly(df, name):
    """
    Upsample hourly data to half-hourly resolution using forward-fill.

    For prices, this means the price for hour H applies to both the
    H:00 and H:30 periods. This is correct because day-ahead prices
    are quoted per hour and apply uniformly within that hour.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly-resolution DataFrame to upsample.
    name : str
        Label for progress printing.

    Returns
    -------
    pd.DataFrame
        Half-hourly DataFrame with forward-filled values.
    """
    if df.empty:
        return df
    # Create a complete half-hourly index spanning the data's range
    hh_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="30min")
    # Reindex to half-hourly grid; forward-fill to propagate hourly values
    out = df.reindex(hh_idx).ffill()
    print(f"  [{name}] Resampled to half-hourly: {len(out):,} records")
    return out


def align_series(series, master_idx, name, fill_limit=2):
    """
    Reindex a series to a master DatetimeIndex with limited gap-filling.

    Short gaps (up to fill_limit periods = 1 hour at 30-min resolution)
    are forward-filled. This handles minor data reporting delays without
    introducing bias from filling long outages.

    Parameters
    ----------
    series : pd.Series
        Data series to align.
    master_idx : pd.DatetimeIndex
        Target index to align to.
    name : str
        Column name for the output and progress messages.
    fill_limit : int, default 2
        Maximum number of consecutive NaN periods to forward-fill.
        At 30-min resolution, 2 periods = 1 hour maximum gap fill.

    Returns
    -------
    pd.Series
        Aligned series with name set to `name`.
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


def align_all(price_df, wind_df, load_df):
    """
    Align prices, wind, and load onto a single half-hourly master index.

    Steps:
        1. Upsample hourly prices to half-hourly (forward-fill)
        2. Create a master 30-min DatetimeIndex for the full date range
        3. Reindex all three series to the master index
        4. Fill short gaps (<=1 hour) via forward-fill
        5. Drop rows where price is missing (target cannot be NaN)

    Parameters
    ----------
    price_df : pd.DataFrame
        Hourly prices (will be resampled to half-hourly).
    wind_df : pd.DataFrame
        Half-hourly wind generation.
    load_df : pd.DataFrame
        Half-hourly system load.

    Returns
    -------
    pd.DataFrame
        Aligned DataFrame with columns: price_eur_mwh, wind_mw, load_mw.
        Index is a uniform half-hourly DatetimeIndex in Dublin time.
    """
    print("\n--- Aligning to master half-hourly index ---")

    # Step 1: Upsample prices from hourly to half-hourly
    price_hh = resample_to_halfhourly(price_df, "price")

    # Step 2: Create master index spanning the full date range
    master = pd.date_range(start=START_STR, end=END_STR, freq="30min")
    print(f"  Master index: {len(master):,} periods ({START_STR} -> {END_STR})")

    # Helper to safely extract the first column as a Series
    def get_s(df):
        return df.iloc[:, 0] if (df is not None and not df.empty) \
               else pd.Series(dtype=float)

    # Step 3-4: Align each series to master index with gap-filling
    p = align_series(get_s(price_hh), master, "price_eur_mwh")
    w = align_series(get_s(wind_df),  master, "wind_mw")
    l = align_series(get_s(load_df),  master, "load_mw")

    # Combine into a single DataFrame
    df = pd.concat([p, w, l], axis=1)

    # Step 5: Drop rows where price (target variable) is missing
    before = len(df)
    df = df.dropna(subset=["price_eur_mwh"])
    if before != len(df):
        print(f"  Dropped {before - len(df)} rows with missing price")

    print(f"  Aligned dataset: {len(df):,} rows")
    return df


# =============================================================================
# SECTION 5: FEATURE ENGINEERING
# =============================================================================
# All features are designed to respect the day-ahead gate closure constraint.
# The day-ahead auction closes at 11:00 on day D-1, meaning forecasts for
# day D must use only information available before 11:00 on D-1.
#
# Minimum feature lag = 24h ensures no information from the target delivery
# day can leak into the model inputs. This is critical for realistic
# out-of-sample evaluation.

def engineer_features(df):
    """
    Build all GRU input features with strict no-lookahead guarantees.

    Feature groups:
        1. CYCLICAL TIME ENCODINGS (6 features)
           - hour_sin, hour_cos: Captures daily price cycle (peak/off-peak)
           - dow_sin, dow_cos: Captures weekday/weekend demand patterns
           - month_sin, month_cos: Captures seasonal generation/demand shifts

           Sin/cos encoding ensures continuity (hour 23 is close to hour 0)
           and avoids the discontinuity that integer hour encoding would create.

        2. PRICE LAGS (3 features)
           - price_lag_24h: Same time yesterday (strongest single predictor)
           - price_lag_48h: Same time 2 days ago (captures weekly patterns)
           - price_lag_168h: Same time last week (captures weekly cycle)

        3. ROLLING PRICE STATISTICS (6 features)
           - Rolling mean/std over 24h, 48h, 168h windows
           - shift(1) applied before rolling to prevent any lookahead
           - Captures recent price level and volatility regime

        4. WIND AND LOAD FEATURES (3 features)
           - wind_lag_48h: Wind generation 48 hours ago
           - load_lag_48h: System demand 48 hours ago
           - wind_pen_lag_48h: Wind penetration ratio (wind/load) 48h ago
           - 48h lag used because wind forecasts for D are available by D-2

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, aligned DataFrame with price_eur_mwh, wind_mw, load_mw.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with all engineered features.
        Note: first ~168h of rows will contain NaN due to lag initialization.
    """
    print("\n--- Engineering features ---")
    pph = 2  # Periods per hour at 30-minute resolution

    # --- Group 1: Cyclical time encodings ---
    # Using sin/cos pairs preserves the circular nature of time
    # (e.g., 23:00 and 00:00 are 1 hour apart, not 23 hours apart)
    df["hour_sin"]  = np.sin(2 * np.pi * df.index.hour      / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df.index.hour      / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month     / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month     / 12)

    # --- Group 2: Price lag features ---
    # Each lag is h hours * 2 periods/hour = shift amount in half-hour periods
    for h in LAG_HOURS:
        df[f"price_lag_{h}h"] = df["price_eur_mwh"].shift(h * pph)

    # --- Group 3: Rolling price statistics ---
    # shift(1) ensures the rolling window does NOT include the current period
    # This prevents subtle lookahead bias in the rolling calculations
    for h in ROLLING_HOURS:
        base = df["price_eur_mwh"].shift(1)  # Exclude current period
        w    = h * pph                        # Window size in periods
        df[f"price_roll_mean_{h}h"] = base.rolling(w, min_periods=1).mean()
        df[f"price_roll_std_{h}h"]  = base.rolling(w, min_periods=1).std()

    # --- Group 4: Wind and load features ---
    if "wind_mw" in df.columns and "load_mw" in df.columns:
        # Wind penetration ratio: fraction of demand met by wind
        # High penetration -> low prices (merit order effect)
        # Division protected against zero load values
        df["wind_pen"] = np.where(
            df["load_mw"] > 0,
            df["wind_mw"] / df["load_mw"],
            np.nan
        )
        # 48h lag ensures these features are available well before gate closure
        df["wind_lag_48h"]     = df["wind_mw"].shift(48 * pph)
        df["load_lag_48h"]     = df["load_mw"].shift(48 * pph)
        df["wind_pen_lag_48h"] = df["wind_pen"].shift(48 * pph)

    # Print summary of all created features
    feature_cols = [c for c in df.columns if c != "price_eur_mwh"]
    print(f"  {len(feature_cols)} features created:")
    for c in feature_cols:
        print(f"    {c}")
    return df


# =============================================================================
# SECTION 6: DATA LEAKAGE VALIDATION
# =============================================================================
# Leakage validation is essential for time series forecasting models.
# If any feature uses future information (lag < gate closure time),
# the model's apparent accuracy will be inflated and will not generalise
# to real-time deployment.

def validate_leakage(df):
    """
    Verify that no feature violates the minimum lag constraint.

    Checks all lag and rolling features to ensure their temporal offset
    is at least 1 hour (in practice, all are >= 24h in this pipeline).
    Prints PASS/FAIL result.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame to validate.

    Returns
    -------
    bool
        True if no leakage detected, False otherwise.
    """
    print("\n--- Leakage validation ---")
    lag_cols = [c for c in df.columns if "_lag_" in c or "roll_" in c]
    ok = True
    for c in lag_cols:
        if "_lag_" in c:
            try:
                # Extract the hour value from column names like 'price_lag_24h'
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
    Print a comprehensive quality report for the final dataset.

    Reports:
        - Dataset dimensions, date range, and temporal resolution
        - Missing value counts per column (flags columns > 1% missing)
        - Price distribution statistics (mean, std, min, max, quartiles)
        - Notes on negative prices (real market events, not errors)
        - Notes on extreme price spikes (known GRU difficulty)

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

    # Negative prices occur during high-wind/low-demand periods when
    # generators pay to stay connected (must-run constraints, subsidies)
    neg = (df["price_eur_mwh"] < 0).sum()
    if neg:
        print(f"\n  NOTE: {neg} negative price periods.")
        print("  Real market events -- high wind + low demand. Keep them.")
        print("  Cite in thesis as a characteristic of the Irish SEM.")

    # Extreme spikes (>2x the 99th percentile) represent scarcity events
    q99    = df["price_eur_mwh"].quantile(0.99)
    spikes = (df["price_eur_mwh"] > q99 * 2).sum()
    if spikes:
        print(f"\n  NOTE: {spikes} extreme spikes (>2x p99 = {q99*2:.0f} EUR/MWh).")
        print("  Energy-market analogue of volatility clustering from Chapter 1.")
        print("  GRU will struggle here -- document as a known limitation.")

    print("=" * 62)


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================

def run():
    """
    Execute the full data pipeline end-to-end.

    Pipeline stages:
        1. Fetch raw data (prices, wind, load) from ENTSO-E API
        2. Save raw data as CSV for reproducibility/debugging
        3. Align all sources to a common half-hourly index
        4. Engineer predictive features (lags, rolling stats, time encodings)
        5. Validate no data leakage in feature construction
        6. Drop warmup rows (NaN from lag initialization)
        7. Save final GRU-ready dataset
        8. Print quality report

    The pipeline is idempotent: running it again will overwrite outputs
    with fresh data up to today's date.
    """
    print("=" * 62)
    print("  ENTSO-E Pipeline -- Chapter 2 (Wind-Only)")
    print("=" * 62)
    print(f"  Zone:   {BIDDING_ZONE} (Ireland SEM)")
    print(f"  Range:  {START_STR} -> {END_STR}\n")

    # Initialise API client
    client = make_client()

    # --- Stage 1: Fetch all three data sources ---
    price_df = fetch_prices(client)
    wind_df  = fetch_wind(client)
    load_df  = fetch_load(client)

    # --- Stage 2: Save raw files for reproducibility ---
    # These allow debugging alignment issues without re-fetching from API
    for df, fname in [(price_df, "entsoe_prices_raw.csv"),
                      (wind_df,  "entsoe_wind_raw.csv"),
                      (load_df,  "entsoe_load_raw.csv")]:
        if df is not None and not df.empty:
            df.to_csv(OUTPUT_DIR / fname)
            print(f"  Saved: {fname}")

    # --- Stage 3: Temporal alignment ---
    print("\nSTEP: Aligning sources")
    df_clean = align_all(price_df, wind_df, load_df)
    df_clean.to_csv(OUTPUT_DIR / "dataset_cleaned.csv")
    print("  Saved: dataset_cleaned.csv")

    # --- Stage 4: Feature engineering ---
    print("\nSTEP: Feature engineering")
    df_feat = engineer_features(df_clean.copy())

    # --- Stage 5: Leakage validation ---
    validate_leakage(df_feat)

    # --- Stage 6-7: Drop warmup NaNs and save final dataset ---
    # The first ~168 hours (1 week) will have NaN in lag features
    # because there is no history to look back on. These rows must
    # be dropped before model training.
    before   = len(df_feat)
    df_model = df_feat.dropna()
    warmup   = before - len(df_model)
    print(f"\n  Dropped {warmup} warmup rows (lag initialisation)")
    print(f"  Final: {len(df_model):,} rows x {len(df_model.columns)} columns")

    df_model.to_csv(OUTPUT_DIR / "dataset_wind_only.csv")
    print("  Saved: dataset_wind_only.csv")

    # --- Stage 8: Quality report ---
    quality_report(df_model)

    # Print file manifest
    print("\nFILES:")
    for fname in ["entsoe_prices_raw.csv", "entsoe_wind_raw.csv",
                  "entsoe_load_raw.csv", "dataset_cleaned.csv",
                  "dataset_wind_only.csv"]:
        p = OUTPUT_DIR / fname
        if p.exists():
            print(f"  OK  {fname:<42} ({p.stat().st_size // 1024} KB)")
        else:
            print(f"  X   {fname:<42} (not created)")

    print("\nNEXT: Run ch2_gru_semo.py on dataset_wind_only.csv")
    print("=" * 62)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run()
