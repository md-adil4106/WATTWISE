"""Data loader module for WATTWISE.

Responsible for ingesting raw electricity consumption datasets, parsing timestamps,
handling missing records, standardizing observations into regular hourly kilowatt (kW) demand,
generating synthetic benchmark demand data, and providing dataset summaries.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DataSummary(dict):
    """Container for dataset summary metadata supporting dict access, attribute access, and tuple unpacking."""

    def __init__(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        n_hours: int,
        missing_rate_before_cleaning: float,
    ):
        super().__init__(
            start=start,
            end=end,
            n_hours=n_hours,
            missing_rate_before_cleaning=missing_rate_before_cleaning,
        )
        self.start = start
        self.end = end
        self.n_hours = n_hours
        self.missing_rate_before_cleaning = missing_rate_before_cleaning

    def __iter__(self):
        yield self["start"]
        yield self["end"]
        yield self["n_hours"]
        yield self["missing_rate_before_cleaning"]


def load_raw(path: Union[str, Path]) -> pd.DataFrame:
    """Load raw electricity consumption time-series data from file.

    Supports:
    (a) UCI household_power_consumption.txt (sep=';', na_values='?', Date d/m/Y and Time, Global_active_power in kW).
    (b) Compatible CSV with either Date+Time columns or a single datetime column plus Global_active_power.

    Args:
        path: Filepath to the raw dataset.

    Returns:
        pd.DataFrame containing the raw measurements.

    Raises:
        ValueError: If file is not found or required columns are missing.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(
            f"Dataset file not found at: '{file_path}'. "
            "Please ensure the file exists or place 'household_power_consumption.txt' inside the 'data/' directory."
        )

    # Detect delimiter by inspecting the header line
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()

    delimiter = ";" if ";" in first_line else ","

    try:
        df = pd.read_csv(
            file_path,
            sep=delimiter,
            na_values=["?", "NA", "null", "None", ""],
            low_memory=False,
        )
    except Exception as exc:
        raise ValueError(f"Failed to read dataset from '{file_path}': {exc}") from exc

    # Column name normalization for case-insensitive check
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    # Verify presence of power column
    target_power_col = None
    for cand in ["global_active_power", "active_power", "power_kw", "kw", "power", "demand_kw"]:
        if cand in cols_lower:
            target_power_col = cols_lower[cand]
            break

    if target_power_col is None:
        raise ValueError(
            f"Missing required power demand column. Expected 'Global_active_power' (or equivalent). "
            f"Available columns: {list(df.columns)}"
        )

    # Ensure consistent column naming for downstream processing
    if target_power_col != "Global_active_power":
        df = df.rename(columns={target_power_col: "Global_active_power"})

    # Verify presence of datetime columns
    has_date_time = ("date" in cols_lower and "time" in cols_lower)
    has_single_dt = any(cand in cols_lower for cand in ["datetime", "timestamp"])

    if not has_date_time and not has_single_dt:
        # Check if index is already datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "Missing required datetime columns. Expected either separate 'Date' and 'Time' columns, "
                f"or a single 'datetime'/'timestamp' column. Available columns: {list(df.columns)}"
            )

    return df


def load_raw_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Alias for load_raw to ensure backwards compatibility with Phase 0 stubs."""
    return load_raw(filepath)


def clean_and_resample(df: pd.DataFrame, freq: str = "1h") -> pd.Series:
    """Build DatetimeIndex, clean invalid records, and resample to hourly average kW demand.

    Steps:
    1. Parse or construct DatetimeIndex.
    2. Coerce Global_active_power to numeric.
    3. Drop invalid values (NaN, negative).
    4. Sort chronologically.
    5. Remove duplicate timestamps (keeping the first occurrence).
    6. Resample to hourly MEAN (kW).
    7. Fill small gaps using time-based interpolation limited to 3 consecutive hours.
    8. Leave longer gaps as NaN and drop them.
    9. Return a pd.Series named 'kw'.

    Args:
        df: Raw DataFrame containing timestamps and Global_active_power.
        freq: Resampling frequency (default '1h').

    Returns:
        pd.Series: Hourly mean electricity demand in kW with DatetimeIndex, named 'kw'.
    """
    df = df.copy()

    # 1. Parse or construct DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        dt_index = df.index
    else:
        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        if "date" in cols_lower and "time" in cols_lower:
            date_col = cols_lower["date"]
            time_col = cols_lower["time"]
            # Fast parsing with dayfirst=True to handle dd/mm/yyyy
            combined = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
            dt_index = pd.to_datetime(combined, format="%d/%m/%Y %H:%M:%S", errors="coerce")
            if dt_index.isna().all():
                # Fallback to general parser if format was not dd/mm/yyyy hh:mm:ss
                dt_index = pd.to_datetime(combined, dayfirst=True, errors="coerce")
        elif "datetime" in cols_lower:
            dt_index = pd.to_datetime(df[cols_lower["datetime"]], errors="coerce")
        elif "timestamp" in cols_lower:
            dt_index = pd.to_datetime(df[cols_lower["timestamp"]], errors="coerce")
        else:
            raise ValueError("Unable to determine datetime column from input DataFrame.")

    # 2. Coerce Global_active_power to numeric
    power_series = pd.to_numeric(df["Global_active_power"], errors="coerce")
    series = pd.Series(power_series.values, index=dt_index)

    # 3. Drop invalid values (NaT index, NaN values, negative values)
    series = series[series.index.notna() & series.notna() & (series >= 0)]

    # 4. Sort chronologically
    series = series.sort_index()

    # 5. Remove duplicate timestamps
    series = series[~series.index.duplicated(keep="first")]

    if series.empty:
        empty_s = pd.Series(dtype="float64", name="kw")
        empty_s.index = pd.DatetimeIndex([], freq=freq)
        return empty_s

    # 6. Resample to hourly MEAN (kW)
    hourly = series.resample(freq).mean()

    # 7 & 8. Fill gaps only by time-based interpolation limited to 3 consecutive hours;
    # leave longer gaps as NaN and then drop them.
    hourly = hourly.interpolate(method="time", limit=3, limit_direction="forward")
    hourly = hourly.dropna()

    # 9. Return series named 'kw'
    hourly.name = "kw"
    return hourly


def generate_synthetic(days: int = 365, seed: int = 42) -> pd.Series:
    """Generate realistic hourly synthetic electricity demand data for testing or demo mode.

    Includes:
    - Daily 24-hour diurnal seasonality (morning and evening peak demand).
    - Weekly cycle (weekday business load vs. weekend residential patterns).
    - Seasonal annual wave (summer cooling and winter heating peaks).
    - Baseload offset and stochastic Gaussian noise.

    Args:
        days: Number of days of hourly synthetic observations (default 365).
        seed: Random state seed for reproducibility.

    Returns:
        pd.Series: Hourly synthetic kW demand series named 'kw', clearly flagged as synthetic.
    """
    rng = np.random.default_rng(seed)
    total_hours = days * 24
    index = pd.date_range(start="2025-01-01 00:00:00", periods=total_hours, freq="1h", name="datetime")

    hours = index.hour.values
    dayofweek = index.day_of_week.values
    dayofyear = index.day_of_year.values

    # Base load level (around 1.5 kW average household load)
    base_load = 1.35

    # Diurnal (24h) pattern: low overnight (02:00-05:00), morning bump (07:00-09:00), evening peak (18:00-21:00)
    morning_peak = 0.65 * np.exp(-((hours - 8) ** 2) / 6.0)
    evening_peak = 1.25 * np.exp(-((hours - 20) ** 2) / 8.0)
    night_trough = -0.35 * np.exp(-((hours - 3.5) ** 2) / 4.0)
    diurnal = morning_peak + evening_peak + night_trough

    # Weekly pattern: slight reduction on weekends or different shift
    is_weekend = (dayofweek >= 5).astype(float)
    weekly_effect = -0.15 * is_weekend + 0.10 * (1 - is_weekend)

    # Annual seasonal pattern: higher in winter (heating) and mid-summer (cooling)
    seasonal_effect = 0.25 * np.cos(2 * np.pi * (dayofyear - 15) / 365.0)

    # Random Gaussian noise
    noise = rng.normal(loc=0.0, scale=0.15, size=total_hours)

    values = base_load + diurnal + weekly_effect + seasonal_effect + noise
    # Demand must be strictly positive
    values = np.clip(values, a_min=0.10, a_max=None)

    synthetic_series = pd.Series(values, index=index, name="kw")
    synthetic_series.attrs["source"] = "SYNTHETIC DEMO"
    synthetic_series.attrs["is_synthetic"] = True
    return synthetic_series


def get_data(
    path: Optional[Union[str, Path]] = None,
    default_uci_path: Union[str, Path] = "data/household_power_consumption.txt",
) -> Tuple[pd.Series, str]:
    """Retrieve electricity demand series from real dataset if available, or fallback to synthetic demo data.

    Attempts to load and clean the dataset from the provided path or default UCI location.
    If the file is absent or cannot be loaded, gracefully falls back to synthetic benchmark data.
    Never labels synthetic data as UCI.

    Args:
        path: Optional explicit file path to load.
        default_uci_path: Default fallback location for UCI dataset.

    Returns:
        Tuple of (pd.Series named 'kw', source_label string).
    """
    target_path = Path(path) if path is not None else Path(default_uci_path)

    if target_path.exists() and target_path.is_file():
        try:
            raw_df = load_raw(target_path)
            clean_series = clean_and_resample(raw_df)
            if not clean_series.empty:
                clean_series.attrs["source"] = "UCI Household Power Consumption"
                clean_series.attrs["is_synthetic"] = False
                return clean_series, "UCI Household Power Consumption"
        except Exception:
            # Fall back to synthetic data if reading fails
            pass

    # Fallback to synthetic demo data
    synthetic_series = generate_synthetic(days=365)
    return synthetic_series, "SYNTHETIC DEMO"


def data_summary(
    series: pd.Series,
    raw_df: Optional[pd.DataFrame] = None,
) -> DataSummary:
    """Generate concise dataset summary metrics.

    Args:
        series: Cleaned hourly electricity demand series.
        raw_df: Optional raw DataFrame to calculate exact pre-cleaning missing rate.

    Returns:
        DataSummary: Summary object containing start, end, n_hours, missing_rate_before_cleaning.
    """
    if series.empty:
        return DataSummary(
            start=pd.NaT,
            end=pd.NaT,
            n_hours=0,
            missing_rate_before_cleaning=1.0,
        )

    start = series.index.min()
    end = series.index.max()
    n_hours = len(series)

    if raw_df is not None and "Global_active_power" in raw_df.columns:
        total_raw = len(raw_df)
        if total_raw > 0:
            numeric_vals = pd.to_numeric(raw_df["Global_active_power"], errors="coerce")
            missing_count = numeric_vals.isna().sum() + (numeric_vals < 0).sum()
            missing_rate = float(missing_count / total_raw)
        else:
            missing_rate = 0.0
    else:
        # Calculate theoretical span in hours
        total_expected_hours = int((end - start) / pd.Timedelta(hours=1)) + 1
        if total_expected_hours > 0:
            missing_rate = float(max(0, total_expected_hours - n_hours) / total_expected_hours)
        else:
            missing_rate = 0.0

    return DataSummary(
        start=start,
        end=end,
        n_hours=n_hours,
        missing_rate_before_cleaning=round(missing_rate, 4),
    )


def validate_data(data: Union[pd.DataFrame, pd.Series]) -> bool:
    """Validate data integrity, schema consistency, missingness ratio, and bounds.

    Args:
        data: Cleaned Series or DataFrame to validate.

    Returns:
        bool: True if validation passes, raises ValueError otherwise.
    """
    if isinstance(data, pd.Series):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Series index must be a pd.DatetimeIndex.")
        if data.isna().any():
            raise ValueError("Cleaned series contains unexpected NaN values.")
        if (data < 0).any():
            raise ValueError("Cleaned series contains invalid negative power values.")
        if not data.index.is_monotonic_increasing:
            raise ValueError("Series index must be strictly monotonic increasing.")
        return True
    elif isinstance(data, pd.DataFrame):
        if "Global_active_power" not in data.columns:
            raise ValueError("DataFrame missing required 'Global_active_power' column.")
        return True
    else:
        raise ValueError(f"Unsupported data type for validation: {type(data)}")
