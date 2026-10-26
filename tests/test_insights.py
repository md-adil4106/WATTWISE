"""Unit tests for WATTWISE operational insights and peak detection."""

import pandas as pd
import pytest

from src.data_loader import generate_synthetic
from src.insights import (
    data_driven_observations,
    hourly_profile,
    peak_summary,
    top_hours,
    weekday_profile,
)


def test_peak_summary():
    """Verify peak_summary correctly flags hours exceeding threshold and identifies peak."""
    timestamps = pd.date_range("2026-01-01 00:00", periods=5, freq="1h")
    forecast_df = pd.DataFrame({
        "timestamp": timestamps,
        "forecast_kw": [1.0, 2.5, 3.8, 1.2, 0.8],
    })

    # Threshold 2.0 kW -> 2 exceeding hours (2.5 and 3.8)
    summary = peak_summary(forecast_df, threshold=2.0)
    assert summary["has_violations"] is True
    assert summary["exceeding_count"] == 2
    assert summary["peak_kw"] == pytest.approx(3.8)
    assert summary["peak_time"] == timestamps[2]
    assert "Peak Alert" in summary["status"]

    # Threshold 4.0 kW -> 0 exceeding hours
    clear_summary = peak_summary(forecast_df, threshold=4.0)
    assert clear_summary["has_violations"] is False
    assert clear_summary["exceeding_count"] == 0
    assert clear_summary["status"] == "No threshold violations predicted"


def test_hourly_profile_shape():
    """Verify hourly profile computes mean demand across all 24 hours of the day."""
    synth = generate_synthetic(days=14, seed=42)
    hp = hourly_profile(synth)

    assert len(hp) == 24
    assert list(hp.index) == list(range(24))
    assert (hp > 0).all()


def test_weekday_profile_shape():
    """Verify weekday profile computes mean demand across all 7 days of the week."""
    synth = generate_synthetic(days=14, seed=42)
    wp = weekday_profile(synth)

    expected_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    assert len(wp) == 7
    assert list(wp.index) == expected_days
    assert (wp > 0).all()


def test_top_hours():
    """Verify top_hours identifies top peak demand hours ordered by rank."""
    synth = generate_synthetic(days=14, seed=42)
    top = top_hours(synth, n=3)

    assert len(top) == 3
    assert list(top.columns) == ["rank", "hour", "avg_kw"]
    assert list(top["rank"]) == [1, 2, 3]
    # Sorted descending by avg_kw
    assert top["avg_kw"].iloc[0] >= top["avg_kw"].iloc[1] >= top["avg_kw"].iloc[2]


def test_data_driven_observations():
    """Verify data_driven_observations produces exactly 3 non-empty observations."""
    synth = generate_synthetic(days=14, seed=42)
    obs = data_driven_observations(synth)

    assert isinstance(obs, list)
    assert len(obs) == 3
    for statement in obs:
        assert isinstance(statement, str)
        assert len(statement) > 20
        assert "kW" in statement
