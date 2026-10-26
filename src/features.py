"""Feature engineering module for WATTWISE.

Responsible for deriving calendar/temporal indicators, historical lag features,
and past rolling statistics for electricity demand forecasting while strictly avoiding
data leakage.
"""

from typing import List, Optional
import pandas as pd


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal calendar features (hour of day, day of week, month, weekend flag).

    Args:
        df: DataFrame with a DatetimeIndex or datetime column.

    Returns:
        pd.DataFrame enriched with calendar feature columns.
    """
    raise NotImplementedError("Stub function: calendar features will be implemented in Phase 2.")


def create_lag_features(df: pd.DataFrame, target_col: str = "demand_kw", lags: Optional[List[int]] = None) -> pd.DataFrame:
    """Construct historical lag features without forward-looking leakage.

    Args:
        df: DataFrame with continuous time series.
        target_col: Column name representing electricity demand.
        lags: List of lag steps (in hours, e.g., [1, 2, 24, 168]).

    Returns:
        pd.DataFrame with added lag feature columns.
    """
    raise NotImplementedError("Stub function: lag feature generation will be implemented in Phase 2.")


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = "demand_kw",
    windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """Calculate rolling window statistics (mean, std, min, max) strictly using prior observations (shifted).

    Args:
        df: DataFrame with continuous time series.
        target_col: Target column name.
        windows: List of window sizes (in hours, e.g., [6, 12, 24]).

    Returns:
        pd.DataFrame with rolling window statistics.
    """
    raise NotImplementedError("Stub function: rolling features will be implemented in Phase 2.")


def prepare_features(df: pd.DataFrame, target_col: str = "demand_kw") -> pd.DataFrame:
    """End-to-end feature pipeline orchestrator combining calendar, lag, and rolling features.

    Args:
        df: Cleaned hourly DataFrame.
        target_col: Name of the electricity demand column.

    Returns:
        pd.DataFrame ready for model training and inference.
    """
    raise NotImplementedError("Stub function: feature preparation pipeline will be implemented in Phase 2.")
