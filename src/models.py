"""Modeling and evaluation module for WATTWISE.

Responsible for temporal train/test splitting, training baseline (Seasonal Naive) and
Random Forest forecasters, computing performance metrics (MAE, RMSE, MAPE), and generating
recursive 24-hour future demand predictions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.features import build_features, make_row


def chrono_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split time-series dataset chronologically without shuffling or lookahead leakage.

    Args:
        df: Feature-engineered DataFrame with DatetimeIndex, sorted chronologically.
        test_frac: Fraction of total observations allocated to the test set (default 0.2).

    Returns:
        Tuple of (train_df, test_df).
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be strictly between 0.0 and 1.0.")

    split_idx = int(len(df) * (1.0 - test_frac))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def train_test_split_temporal(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Alias for chrono_split supporting Phase 0 API."""
    return chrono_split(df, test_frac=test_size)


def seasonal_naive(
    series: pd.Series,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Predict demand for timestamp t using the historical observation at t - 24 hours (daily persistence).

    Args:
        series: Full historical hourly electricity demand series named 'kw'.
        index: Target timestamps for which predictions are required.

    Returns:
        pd.Series: Baseline predictions indexed by index.
    """
    preds = []
    for ts in index:
        target_lag_ts = ts - pd.Timedelta(hours=24)
        if target_lag_ts in series.index:
            preds.append(float(series.loc[target_lag_ts]))
        else:
            # Fallback to closest available prior observation
            prior = series[series.index <= target_lag_ts]
            if not prior.empty:
                preds.append(float(prior.iloc[-1]))
            else:
                preds.append(float(series.iloc[0]))

    return pd.Series(preds, index=index, name="seasonal_naive")


def train_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train a Random Forest Regressor on tabular lag and calendar features.

    Specifications:
    - n_estimators=200
    - max_depth=12
    - n_jobs=-1
    - random_state=42

    Args:
        X_train: Training features.
        y_train: Training target demand in kW.
        random_state: Seed for reproducibility.

    Returns:
        Fitted RandomForestRegressor model.
    """
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    return rf


def train_forecaster(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> RandomForestRegressor:
    """Alias for train_rf supporting Phase 0 API."""
    return train_rf(X_train, y_train)


def metrics(
    y_true: Union[np.ndarray, pd.Series, List[float]],
    y_pred: Union[np.ndarray, pd.Series, List[float]],
) -> Dict[str, float]:
    """Compute standard forecasting evaluation metrics: MAE, RMSE, and conditional MAPE.

    Note on MAPE:
    Electricity load values near zero produce extreme or undefined relative percentages.
    Following industrial best practice, MAPE is computed strictly on observations where
    actual demand >= 0.1 kW to avoid division-by-zero artifacts.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dict with 'MAE', 'RMSE', and 'MAPE' keys.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if len(y_t) == 0:
        return {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0}

    mae = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

    # Compute MAPE conditionally on actual values >= 0.1 kW
    valid_mask = y_t >= 0.1
    if np.any(valid_mask):
        mape = float(np.mean(np.abs((y_t[valid_mask] - y_p[valid_mask]) / y_t[valid_mask])) * 100.0)
    else:
        mape = 0.0

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 2),
    }


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Alias for metrics supporting Phase 0 API."""
    return metrics(y_true, y_pred)


def evaluate(
    series: pd.Series,
    test_frac: float = 0.2,
) -> Dict[str, Any]:
    """Evaluate both Seasonal Naive and Random Forest models on identical test timestamps.

    Workflow:
    1. Build feature matrix with build_features(series).
    2. Split chronologically using chrono_split (test_frac=0.2).
    3. Fit Random Forest model on training set.
    4. Predict on identical test timestamps for both models.
    5. Compute MAE, RMSE, and MAPE for both models.
    6. Determine 'best_model' dynamically from lowest RMSE (never hardcoded).

    Args:
        series: Hourly electricity demand series named 'kw'.
        test_frac: Fraction of data for test evaluation.

    Returns:
        Dict containing:
        - 'test_index': test DatetimeIndex
        - 'y_true': ground truth test values (np.ndarray)
        - 'baseline_preds': seasonal naive test predictions (np.ndarray)
        - 'rf_preds': random forest test predictions (np.ndarray)
        - 'baseline_metrics': dict of MAE, RMSE, MAPE
        - 'rf_metrics': dict of MAE, RMSE, MAPE
        - 'best_model': name of the superior model by RMSE
        - 'model': trained Random Forest model object
        - 'feature_cols': list of feature names used
    """
    df = build_features(series)
    train_df, test_df = chrono_split(df, test_frac=test_frac)

    feature_cols = [c for c in df.columns if c != "y"]
    X_train, y_train = train_df[feature_cols], train_df["y"]
    X_test, y_test = test_df[feature_cols], test_df["y"]

    # 1. Train ML model
    rf = train_rf(X_train, y_train)
    rf_preds = rf.predict(X_test)

    # 2. Evaluate Baseline on identical test timestamps
    baseline_series = seasonal_naive(series, test_df.index)
    baseline_preds = baseline_series.values

    # 3. Compute Metrics
    rf_metrics = metrics(y_test.values, rf_preds)
    baseline_metrics = metrics(y_test.values, baseline_preds)

    # 4. Compute best model from lowest RMSE (dynamically computed, never hardcoded)
    best_model = "Random Forest" if rf_metrics["RMSE"] < baseline_metrics["RMSE"] else "Seasonal Naive"

    return {
        "test_index": test_df.index,
        "y_true": y_test.values,
        "baseline_preds": baseline_preds,
        "rf_preds": rf_preds,
        "baseline_metrics": baseline_metrics,
        "rf_metrics": rf_metrics,
        "best_model": best_model,
        "model": rf,
        "feature_cols": feature_cols,
    }


def forecast_next_24(
    model: Any,
    series: pd.Series,
) -> pd.DataFrame:
    """Generate recursive 24-step hourly electricity demand forecasts.

    Workflow:
    - Forecast starts strictly at: last observation timestamp + 1 hour.
    - At each step, builds features with make_row using past history.
    - Earlier predictions are appended to history for subsequent multi-step lags.
    - Repeats recursively 24 times.

    Args:
        model: Fitted forecaster (e.g. trained RandomForestRegressor).
        series: Historical hourly electricity demand series.

    Returns:
        pd.DataFrame with 'timestamp' and 'forecast_kw' columns.
    """
    if series.empty:
        raise ValueError("Cannot forecast from an empty series.")

    last_ts = series.index[-1]
    feature_cols = getattr(model, "feature_names_in_", None)
    if feature_cols is not None:
        feature_cols = list(feature_cols)

    history = series.copy()
    timestamps = []
    forecast_values = []

    current_ts = last_ts + pd.Timedelta(hours=1)

    for _ in range(24):
        # Build features for current_ts using only values strictly before current_ts
        row = make_row(history, current_ts, feature_cols=feature_cols)

        pred_val = float(model.predict(row)[0])
        # Demand is non-negative
        pred_val = max(0.0, round(pred_val, 4))

        timestamps.append(current_ts)
        forecast_values.append(pred_val)

        # Append prediction to history for next recursive steps
        step_series = pd.Series([pred_val], index=[current_ts])
        history = pd.concat([history, step_series])

        current_ts += pd.Timedelta(hours=1)

    forecast_df = pd.DataFrame({
        "timestamp": timestamps,
        "forecast_kw": forecast_values,
    })
    return forecast_df


def generate_forecast(model: Any, future_features: pd.DataFrame) -> np.ndarray:
    """Alias for model.predict supporting Phase 0 API."""
    return model.predict(future_features)
