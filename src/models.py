"""Modeling and evaluation module for WATTWISE.

Responsible for temporal train/test splitting, training baseline and machine learning
forecasters (e.g., Ridge, Random Forest, Gradient Boosting), computing performance metrics
(MAE, RMSE, MAPE, R2), and generating future demand predictions.
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


def train_test_split_temporal(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split time-series data chronologically to preserve temporal order (strictly no shuffling).

    Args:
        df: Feature-engineered DataFrame sorted by time.
        test_size: Proportion of the dataset to allocate to the test set (most recent chunk).

    Returns:
        Tuple of (train_df, test_df).
    """
    raise NotImplementedError("Stub function: temporal split will be implemented in Phase 3.")


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Train a simple baseline model (e.g., historical persistence or Ridge regression).

    Args:
        X_train: Training features.
        y_train: Training target demand.

    Returns:
        Fitted baseline model.
    """
    raise NotImplementedError("Stub function: baseline model will be implemented in Phase 3.")


def train_forecaster(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest"
) -> Any:
    """Train a primary tabular forecasting model (e.g., Random Forest or Gradient Boosting).

    Args:
        X_train: Training features.
        y_train: Training target demand.
        model_type: Model architecture key.

    Returns:
        Fitted forecasting model.
    """
    raise NotImplementedError("Stub function: forecasting model training will be implemented in Phase 3.")


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard forecasting evaluation metrics (MAE, RMSE, MAPE, R2).

    Args:
        y_true: True ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary of metric names and their calculated values.
    """
    raise NotImplementedError("Stub function: evaluation metrics will be implemented in Phase 3.")


def generate_forecast(model: Any, future_features: pd.DataFrame) -> np.ndarray:
    """Generate electricity demand forecasts for upcoming time horizons.

    Args:
        model: Fitted forecasting model.
        future_features: DataFrame containing features for target prediction horizon.

    Returns:
        np.ndarray containing predicted kW demand.
    """
    raise NotImplementedError("Stub function: forecast generation will be implemented in Phase 3.")
