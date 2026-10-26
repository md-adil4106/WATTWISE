"""Unit tests for WATTWISE data loading, datetime parsing, and hourly resampling."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    clean_and_resample,
    data_summary,
    generate_synthetic,
    get_data,
    load_raw,
    validate_data,
)


def test_missing_file_error():
    """Verify load_raw raises ValueError with a clear message when file is missing."""
    missing_path = "data/non_existent_file_12345.txt"
    with pytest.raises(ValueError, match="not found"):
        load_raw(missing_path)


def test_question_mark_and_invalid_handling():
    """Verify that '?' and invalid values in Global_active_power are coerced to NaN and dropped."""
    raw_df = pd.DataFrame({
        "Date": ["01/01/2025", "01/01/2025", "01/01/2025", "01/01/2025"],
        "Time": ["10:00:00", "10:15:00", "10:30:00", "10:45:00"],
        "Global_active_power": ["2.0", "?", "-1.5", "4.0"],
    })
    res = clean_and_resample(raw_df)
    # The valid values are 2.0 and 4.0; average for 10:00:00 hour should be (2.0 + 4.0) / 2 = 3.0 kW
    assert len(res) == 1
    assert res.name == "kw"
    assert res.iloc[0] == pytest.approx(3.0)


def test_duplicate_timestamps_removal():
    """Verify that duplicate timestamps are removed, keeping the first observation."""
    raw_df = pd.DataFrame({
        "Date": ["01/01/2025", "01/01/2025", "01/01/2025"],
        "Time": ["10:00:00", "10:00:00", "10:30:00"],
        "Global_active_power": [2.0, 99.0, 4.0],  # 99.0 is duplicate at 10:00:00
    })
    res = clean_and_resample(raw_df)
    # 10:00:00 should use 2.0 (not 99.0), combined with 10:30:00 (4.0) -> mean = 3.0 kW
    assert len(res) == 1
    assert res.iloc[0] == pytest.approx(3.0)


def test_hourly_resampling_mean_handcrafted():
    """Verify hourly resampling computes exact mean on a handcrafted multi-observation schedule."""
    raw_df = pd.DataFrame({
        "Date": [
            "01/01/2025", "01/01/2025", "01/01/2025", "01/01/2025",  # Hour 10
            "01/01/2025", "01/01/2025",                              # Hour 11
        ],
        "Time": [
            "10:00:00", "10:15:00", "10:30:00", "10:45:00",
            "11:00:00", "11:30:00",
        ],
        "Global_active_power": [
            1.0, 2.0, 3.0, 4.0,  # Mean = 2.5
            4.0, 6.0,            # Mean = 5.0
        ],
    })
    res = clean_and_resample(raw_df)

    assert isinstance(res, pd.Series)
    assert res.name == "kw"
    assert len(res) == 2
    assert res.loc["2025-01-01 10:00:00"] == pytest.approx(2.5)
    assert res.loc["2025-01-01 11:00:00"] == pytest.approx(5.0)


def test_monotonic_increasing_index():
    """Verify that input with unordered timestamps is sorted to a strictly monotonic increasing index."""
    raw_df = pd.DataFrame({
        "Date": ["01/01/2025", "01/01/2025", "01/01/2025"],
        "Time": ["14:00:00", "10:00:00", "12:00:00"],  # Unordered
        "Global_active_power": [3.0, 1.0, 2.0],
    })
    res = clean_and_resample(raw_df)

    assert res.index.is_monotonic_increasing
    # Hours 10, 12, 14 have 1-hour gaps at 11 and 13 which are interpolated
    assert list(res.index.hour) == [10, 11, 12, 13, 14]
    assert res.loc["2025-01-01 10:00:00"] == pytest.approx(1.0)
    assert res.loc["2025-01-01 11:00:00"] == pytest.approx(1.5)
    assert res.loc["2025-01-01 12:00:00"] == pytest.approx(2.0)
    assert res.loc["2025-01-01 13:00:00"] == pytest.approx(2.5)
    assert res.loc["2025-01-01 14:00:00"] == pytest.approx(3.0)
    assert validate_data(res) is True


def test_time_interpolation_and_gap_handling():
    """Verify that gaps <= 3 hours are interpolated, while longer gaps leave NaNs that get dropped."""
    raw_df = pd.DataFrame({
        "Date": ["01/01/2025", "01/01/2025"],
        "Time": ["01:00:00", "04:00:00"],  # Gap of 2 missing hours: 02:00 and 03:00
        "Global_active_power": [1.0, 4.0],
    })
    res = clean_and_resample(raw_df)
    # Between 01:00 (1.0) and 04:00 (4.0), 2 hours (02:00 -> 2.0, 03:00 -> 3.0) should be interpolated
    assert len(res) == 4
    assert res.loc["2025-01-01 02:00:00"] == pytest.approx(2.0)
    assert res.loc["2025-01-01 03:00:00"] == pytest.approx(3.0)


def test_synthetic_labelling():
    """Verify that synthetic generation is clearly labelled and never falsely attributed to UCI."""
    # Direct synthetic generation
    synth = generate_synthetic(days=14, seed=123)
    assert isinstance(synth, pd.Series)
    assert synth.name == "kw"
    assert synth.attrs.get("source") == "SYNTHETIC DEMO"
    assert synth.attrs.get("is_synthetic") is True
    assert (synth > 0).all()

    # get_data fallback with non-existent path
    fallback_series, label = get_data(path="non_existent_dataset.txt")
    assert label == "SYNTHETIC DEMO"
    assert "UCI" not in label
    assert fallback_series.attrs.get("is_synthetic") is True


def test_data_summary():
    """Verify that data_summary returns start, end, n_hours, and missing_rate_before_cleaning."""
    synth = generate_synthetic(days=7)
    summary = data_summary(synth)

    # Test dict-style access
    assert "start" in summary
    assert "end" in summary
    assert summary["n_hours"] == 7 * 24
    assert summary["missing_rate_before_cleaning"] == 0.0

    # Test attribute access
    assert summary.n_hours == 168

    # Test tuple unpacking
    start, end, n_hours, missing_rate = summary
    assert n_hours == 168
    assert missing_rate == 0.0
