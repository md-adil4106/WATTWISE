"""Insights and analytics module for WATTWISE.

Responsible for computing operational energy insights, detecting peak demand periods,
analyzing hourly and weekly consumption profiles, and generating data-driven observations.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def peak_summary(forecast_df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    """Analyze 24-hour forecasted demand against an operational peak threshold.

    Args:
        forecast_df: DataFrame with 'timestamp' and 'forecast_kw' columns.
        threshold: Power threshold in kW above which peaks are flagged.

    Returns:
        Dict containing:
        - 'exceeding_df': rows in forecast_df where forecast_kw > threshold
        - 'exceeding_count': number of hours exceeding the threshold
        - 'peak_time': timestamp of the highest forecasted kW
        - 'peak_kw': highest forecasted power in kW
        - 'status': status string ('No threshold violations predicted' or peak alert message)
        - 'has_violations': boolean flag
        - 'threshold': evaluated threshold
    """
    if forecast_df.empty or "forecast_kw" not in forecast_df.columns:
        return {
            "exceeding_df": pd.DataFrame(columns=["timestamp", "forecast_kw"]),
            "exceeding_count": 0,
            "peak_time": None,
            "peak_kw": 0.0,
            "status": "No forecast data available",
            "has_violations": False,
            "threshold": threshold,
        }

    exceeding = forecast_df[forecast_df["forecast_kw"] > threshold].copy()
    max_idx = forecast_df["forecast_kw"].idxmax()
    peak_row = forecast_df.loc[max_idx]
    peak_time = peak_row["timestamp"]
    peak_kw = float(peak_row["forecast_kw"])

    has_violations = len(exceeding) > 0
    if has_violations:
        status = f"Peak Alert: {len(exceeding)} hour(s) exceed threshold of {threshold:.2f} kW"
    else:
        status = "No threshold violations predicted"

    return {
        "exceeding_df": exceeding,
        "exceeding_count": len(exceeding),
        "peak_time": peak_time,
        "peak_kw": round(peak_kw, 4),
        "status": status,
        "has_violations": has_violations,
        "threshold": round(threshold, 4),
    }


def hourly_profile(series: pd.Series) -> pd.Series:
    """Compute average hourly electricity demand across the 24 hours of the day (0-23).

    Args:
        series: Hourly electricity demand series named 'kw' with DatetimeIndex.

    Returns:
        pd.Series: Hourly mean demand indexed by hour of day (0..23).
    """
    profile = series.groupby(series.index.hour).mean()
    profile.name = "hourly_avg_kw"
    profile.index.name = "hour"
    return profile


def weekday_profile(series: pd.Series) -> pd.Series:
    """Compute average electricity demand for each day of the week (Monday through Sunday).

    Args:
        series: Hourly electricity demand series with DatetimeIndex.

    Returns:
        pd.Series: Average kW indexed by day name (Monday..Sunday).
    """
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_indices = series.index.day_of_week
    profile = series.groupby(day_indices).mean()
    profile.index = [day_names[i] for i in profile.index]
    profile.name = "weekday_avg_kw"
    profile.index.name = "day_of_week"
    return profile


def top_hours(series: pd.Series, n: int = 3) -> pd.DataFrame:
    """Identify the top n hours of the day with the highest average electricity demand.

    Args:
        series: Hourly electricity demand series.
        n: Number of top peak hours to return (default 3).

    Returns:
        pd.DataFrame: Table of top hours with columns ['rank', 'hour', 'avg_kw'].
    """
    hp = series.groupby(series.index.hour).mean().reset_index()
    hp.columns = ["hour", "avg_kw"]
    top = hp.sort_values(by="avg_kw", ascending=False).head(n).copy()
    top["avg_kw"] = top["avg_kw"].round(3)
    top["rank"] = range(1, len(top) + 1)
    return top[["rank", "hour", "avg_kw"]].reset_index(drop=True)


def data_driven_observations(series: pd.Series) -> List[str]:
    """Formulate 3 concise, data-driven operational observations from the dataset.

    Computed dynamically from the actual time-series observations:
    1. Daily peak hour window and deviation above baseline.
    2. Weekend vs. weekday consumption differential.
    3. Overnight baseload trough timing and magnitude.

    Args:
        series: Cleaned hourly electricity demand series.

    Returns:
        List of 3 formatted observation strings.
    """
    if series.empty:
        return [
            "Insufficient data to compute operational observations.",
            "Historical consumption profile is currently empty.",
            "Please load a valid dataset.",
        ]

    hp = series.groupby(series.index.hour).mean()
    peak_h = int(hp.idxmax())
    peak_val = float(hp[peak_h])
    overall_mean = float(series.mean())
    peak_pct = ((peak_val - overall_mean) / overall_mean) * 100.0 if overall_mean > 0 else 0.0

    # 1. Peak demand observation
    obs_peak = (
        f"Peak electricity demand concentrates at {peak_h:02d}:00 (averaging {peak_val:.2f} kW, "
        f"{peak_pct:+.1f}% relative to the overall mean of {overall_mean:.2f} kW)."
    )

    # 2. Weekday vs weekend observation
    weekday_s = series[series.index.day_of_week < 5]
    weekend_s = series[series.index.day_of_week >= 5]
    weekday_avg = float(weekday_s.mean()) if not weekday_s.empty else overall_mean
    weekend_avg = float(weekend_s.mean()) if not weekend_s.empty else overall_mean

    if weekday_avg > 0:
        diff_pct = ((weekend_avg - weekday_avg) / weekday_avg) * 100.0
        dir_text = "higher" if diff_pct >= 0 else "lower"
        obs_weekend = (
            f"Weekends average {abs(diff_pct):.1f}% {dir_text} demand than weekdays "
            f"(avg {weekend_avg:.2f} kW on weekends vs. {weekday_avg:.2f} kW on weekdays)."
        )
    else:
        obs_weekend = f"Weekend demand averages {weekend_avg:.2f} kW compared to weekday demand of {weekday_avg:.2f} kW."

    # 3. Overnight baseload observation
    trough_h = int(hp.idxmin())
    trough_val = float(hp[trough_h])
    obs_trough = (
        f"Minimum baseload occurs at {trough_h:02d}:00 (averaging {trough_val:.2f} kW), "
        "representing quiet-hour standby power consumption."
    )

    return [obs_peak, obs_weekend, obs_trough]


# Backwards compatibility helpers
def compute_peak_hours(df: Any, **kwargs) -> Dict[str, Any]:
    """Compatibility wrapper for peak identification."""
    if isinstance(df, pd.Series):
        s = df
    elif isinstance(df, pd.DataFrame) and "Global_active_power" in df.columns:
        s = df["Global_active_power"]
    elif isinstance(df, pd.DataFrame) and "kw" in df.columns:
        s = df["kw"]
    else:
        s = pd.Series(dtype=float)
    return {"top_hours": top_hours(s) if not s.empty else []}


def detect_anomalies(df: Any, **kwargs) -> pd.DataFrame:
    """Compatibility wrapper for anomaly detection."""
    return pd.DataFrame()


def generate_summary_insights(df: Any, forecast_df: Any, metrics: Any) -> Dict[str, Any]:
    """Compatibility wrapper for summary generation."""
    return {}
