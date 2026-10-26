"""Feature engineering module for WATTWISE.

Responsible for deriving calendar/temporal indicators, historical lag features,
and past rolling statistics for electricity demand forecasting with zero data leakage.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd


def create_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Extract temporal calendar features (hour of day, day of week) from a DatetimeIndex.

    Args:
        index: DatetimeIndex of the series.

    Returns:
        pd.DataFrame with 'hour' and 'dayofweek' columns.
    """
    df = pd.DataFrame(index=index)
    df["hour"] = index.hour
    df["dayofweek"] = index.day_of_week
    return df


def create_lag_features(
    series: pd.Series,
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Construct historical lag features without forward-looking leakage.

    Args:
        series: Continuous hourly electricity demand series.
        lags: List of lag steps in hours. Default: [1, 2, 3, 24, 168].

    Returns:
        pd.DataFrame with lag feature columns.
    """
    if lags is None:
        lags = [1, 2, 3, 24]
        if len(series) > 168:
            lags.append(168)

    df = pd.DataFrame(index=series.index)
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    return df


def create_rolling_features(
    series: pd.Series,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Calculate rolling window statistics strictly using prior observations (shifted by 1).

    Args:
        series: Continuous hourly electricity demand series.
        windows: List of window sizes in hours. Default: [24, 168].

    Returns:
        pd.DataFrame with rolling mean columns.
    """
    if windows is None:
        windows = [24]
        if len(series) > 168:
            windows.append(168)

    df = pd.DataFrame(index=series.index)
    # series.shift(1) ensures the current hour's value is NEVER included in rolling statistics
    shifted = series.shift(1)
    for win in windows:
        df[f"roll_mean_{win}"] = shifted.rolling(win).mean()
    return df


def build_features(series: pd.Series) -> pd.DataFrame:
    """Build feature matrix with target 'y' from an hourly electricity demand series.

    Features generated:
    - lag_1: value at t-1
    - lag_2: value at t-2
    - lag_3: value at t-3
    - lag_24: value at t-24
    - lag_168: value at t-168 (if series length > 168)
    - roll_mean_24: 24-hour rolling mean computed on series.shift(1) (zero leakage)
    - roll_mean_168: 168-hour rolling mean computed on series.shift(1) (if series length > 168)
    - hour: hour of day (0-23)
    - dayofweek: day of week (0-6)
    - y: current hour's kw target

    Args:
        series: Hourly electricity demand pd.Series named 'kw' with DatetimeIndex.

    Returns:
        pd.DataFrame: Clean feature matrix with all NaN lag rows dropped.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a DatetimeIndex.")

    include_168 = len(series) > 168

    df = pd.DataFrame(index=series.index)
    df["lag_1"] = series.shift(1)
    df["lag_2"] = series.shift(2)
    df["lag_3"] = series.shift(3)
    df["lag_24"] = series.shift(24)

    if include_168:
        df["lag_168"] = series.shift(168)

    # Strictly shifted: excludes current hour
    df["roll_mean_24"] = series.shift(1).rolling(24).mean()

    if include_168:
        df["roll_mean_168"] = series.shift(1).rolling(168).mean()

    df["hour"] = series.index.hour
    df["dayofweek"] = series.index.day_of_week

    # Target: current hour's actual kw
    df["y"] = series.values

    # Drop initial rows containing NaN from lag shifts
    df = df.dropna()
    return df


def make_row(
    history_series: pd.Series,
    timestamp: Union[pd.Timestamp, str],
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Construct a single feature row for ONE future timestamp using strictly prior observations.

    Shared by training evaluation and recursive multi-step forecasting so logic is identical.
    Uses only history where timestamp' < timestamp.

    Args:
        history_series: Series containing historical observations (and earlier predictions).
        timestamp: The future timestamp for which features are to be built.
        feature_cols: Optional explicit list of feature columns to align with model expectations.

    Returns:
        pd.DataFrame: 1-row DataFrame indexed by timestamp containing all feature columns.
    """
    ts = pd.Timestamp(timestamp)
    # Strictly filter prior values (strictly before timestamp)
    past = history_series[history_series.index < ts]

    if past.empty:
        raise ValueError(f"No historical observations available strictly prior to {ts}.")

    def _get_lag(past_s: pd.Series, target_ts: pd.Timestamp, lag_hours: int) -> float:
        expected_ts = target_ts - pd.Timedelta(hours=lag_hours)
        if expected_ts in past_s.index:
            return float(past_s.loc[expected_ts])
        # Position-based fallback from tail if continuous
        if len(past_s) >= lag_hours:
            return float(past_s.iloc[-lag_hours])
        return float(past_s.iloc[0])

    include_168 = (
        (feature_cols is not None and "lag_168" in feature_cols)
        or (feature_cols is None and len(past) >= 168)
    )

    row = {
        "lag_1": _get_lag(past, ts, 1),
        "lag_2": _get_lag(past, ts, 2),
        "lag_3": _get_lag(past, ts, 3),
        "lag_24": _get_lag(past, ts, 24),
    }

    if include_168:
        row["lag_168"] = _get_lag(past, ts, 168)

    # 24h rolling mean strictly before ts
    window_24 = past.iloc[-24:] if len(past) >= 24 else past
    row["roll_mean_24"] = float(window_24.mean())

    if include_168:
        window_168 = past.iloc[-168:] if len(past) >= 168 else past
        row["roll_mean_168"] = float(window_168.mean())

    row["hour"] = int(ts.hour)
    row["dayofweek"] = int(ts.day_of_week)

    row_df = pd.DataFrame([row], index=[ts])

    if feature_cols is not None:
        row_df = row_df[feature_cols]

    return row_df


def prepare_features(series: pd.Series) -> pd.DataFrame:
    """Alias for build_features to support Phase 0 API."""
    return build_features(series)
