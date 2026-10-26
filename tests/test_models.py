"""Unit tests for WATTWISE feature engineering, models, chronological split, and forecasting."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_synthetic
from src.features import build_features, make_row
from src.models import (
    chrono_split,
    evaluate,
    forecast_next_24,
    metrics,
    seasonal_naive,
    train_rf,
)


def test_no_leakage():
    """Verify features at timestamp t are completely unchanged if values at and after t are altered."""
    series = generate_synthetic(days=14, seed=42)
    f1 = build_features(series)
    feature_cols = [c for c in f1.columns if c != "y"]

    # Choose a reference timestamp well inside the series
    t_ref = f1.index[50]

    # Alter history: replace all values at and after t_ref with extreme noise
    series_altered = series.copy()
    future_or_current_mask = series_altered.index >= t_ref
    series_altered.loc[future_or_current_mask] = np.random.uniform(500.0, 1000.0, size=future_or_current_mask.sum())

    f2 = build_features(series_altered)

    # Features at t_ref must be 100% identical because they depend strictly on t < t_ref
    pd.testing.assert_series_equal(f1.loc[t_ref, feature_cols], f2.loc[t_ref, feature_cols])

    # Also verify make_row matches build_features on the unaltered series
    row = make_row(series, t_ref, feature_cols=feature_cols)
    pd.testing.assert_series_equal(
        f1.loc[t_ref, feature_cols],
        row.iloc[0],
        check_names=False,
    )


def test_chronological_split_no_overlap():
    """Verify that chrono_split produces strictly chronological partitions with zero temporal overlap."""
    series = generate_synthetic(days=14, seed=42)
    df = build_features(series)

    train_df, test_df = chrono_split(df, test_frac=0.25)

    assert len(train_df) + len(test_df) == len(df)
    assert len(test_df) == int(len(df) * 0.25)

    # Strictly chronological: maximum train timestamp must precede minimum test timestamp
    assert train_df.index.max() < test_df.index.min()

    # Both partitions must be monotonically increasing
    assert train_df.index.is_monotonic_increasing
    assert test_df.index.is_monotonic_increasing


def test_forecast_24_hours_shape_and_timeline():
    """Verify recursive forecast produces exactly 24 hourly rows starting 1h after the latest observation."""
    series = generate_synthetic(days=14, seed=42)
    df = build_features(series)
    feature_cols = [c for c in df.columns if c != "y"]

    rf = train_rf(df[feature_cols].iloc[:100], df["y"].iloc[:100], random_state=42)

    forecast_df = forecast_next_24(rf, series)

    # Exactly 24 steps
    assert len(forecast_df) == 24
    assert list(forecast_df.columns) == ["timestamp", "forecast_kw"]

    # Start timestamp must be strictly 1 hour after the last series observation
    expected_start = series.index[-1] + pd.Timedelta(hours=1)
    assert forecast_df["timestamp"].iloc[0] == expected_start

    # Verify hourly timeline spacing
    diffs = forecast_df["timestamp"].diff().dropna()
    assert (diffs == pd.Timedelta(hours=1)).all()

    # Forecast values must be strictly valid numbers
    assert not forecast_df["forecast_kw"].isna().any()
    assert (forecast_df["forecast_kw"] >= 0).all()


def test_metrics_on_known_array():
    """Verify metrics computes exact MAE, RMSE, and conditional MAPE on a known numerical array."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.0, 2.5, 4.0])

    # Absolute errors: [0.5, 0.0, 0.5, 0.0] -> MAE = 0.25
    # Squared errors: [0.25, 0.0, 0.25, 0.0] -> RMSE = sqrt(0.5 / 4) = sqrt(0.125) ≈ 0.3536
    # Relative errors: [0.5/1.0, 0.0, 0.5/3.0, 0.0] -> [0.5, 0.0, 0.166667, 0.0] -> MAPE ≈ 16.67%
    res = metrics(y_true, y_pred)

    assert res["MAE"] == pytest.approx(0.25, abs=1e-3)
    assert res["RMSE"] == pytest.approx(0.3536, abs=1e-3)
    assert res["MAPE"] == pytest.approx(16.67, abs=1e-2)


def test_seasonal_naive_equals_lag_24():
    """Verify that the seasonal naive baseline prediction for timestamp t equals the value at t - 24h."""
    series = generate_synthetic(days=14, seed=42)
    df = build_features(series)

    # Predict baseline on df.index
    baseline_series = seasonal_naive(series, df.index)

    # Must match df['lag_24'] exactly
    np.testing.assert_allclose(baseline_series.values, df["lag_24"].values)


def test_evaluate_dynamic_selection():
    """Verify evaluate runs end-to-end and selects best model dynamically by lowest RMSE."""
    series = generate_synthetic(days=14, seed=42)
    eval_result = evaluate(series, test_frac=0.2)

    assert "test_index" in eval_result
    assert "y_true" in eval_result
    assert "baseline_preds" in eval_result
    assert "rf_preds" in eval_result
    assert "baseline_metrics" in eval_result
    assert "rf_metrics" in eval_result
    assert "best_model" in eval_result

    # Evaluated on identical test timestamps
    n_test = len(eval_result["test_index"])
    assert len(eval_result["y_true"]) == n_test
    assert len(eval_result["baseline_preds"]) == n_test
    assert len(eval_result["rf_preds"]) == n_test

    # Best model selection matches RMSE
    rf_rmse = eval_result["rf_metrics"]["RMSE"]
    base_rmse = eval_result["baseline_metrics"]["RMSE"]
    expected_best = "Random Forest" if rf_rmse < base_rmse else "Seasonal Naive"
    assert eval_result["best_model"] == expected_best
