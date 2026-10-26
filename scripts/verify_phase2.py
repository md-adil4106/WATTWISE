"""Verification script for WATTWISE Phase 2 feature engineering, models, and forecasting."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from src.data_loader import get_data
from src.models import evaluate, forecast_next_24


def main():
    print("=" * 65)
    print("WATTWISE Phase 2 Modeling & Forecasting Verification")
    print("=" * 65)

    t0 = time.time()
    series, source_label = get_data()
    print(f"Dataset Source:            {source_label}")
    print(f"Total Hourly Observations: {len(series):,}")
    print(f"Date Range:                {series.index[0]} to {series.index[-1]}")
    print("-" * 65)

    print("Training & Evaluating Models (Seasonal Naive vs. Random Forest)...")
    eval_res = evaluate(series, test_frac=0.2)
    t1 = time.time()
    print(f"Evaluation completed in {round(t1 - t0, 2)} seconds")
    print("-" * 65)

    print(f"Test Set Observations:     {len(eval_res['test_index']):,}")
    print(f"Test Date Range:           {eval_res['test_index'][0]} to {eval_res['test_index'][-1]}")
    print()
    print("Performance Metrics:")
    print("  [Baseline - Seasonal Naive]")
    print(f"    MAE:  {eval_res['baseline_metrics']['MAE']:.4f} kW")
    print(f"    RMSE: {eval_res['baseline_metrics']['RMSE']:.4f} kW")
    print(f"    MAPE: {eval_res['baseline_metrics']['MAPE']:.2f}%")
    print("  [Model - Random Forest]")
    print(f"    MAE:  {eval_res['rf_metrics']['MAE']:.4f} kW")
    print(f"    RMSE: {eval_res['rf_metrics']['RMSE']:.4f} kW")
    print(f"    MAPE: {eval_res['rf_metrics']['MAPE']:.2f}%")
    print()
    print(f"Superior Model by RMSE:    >>> {eval_res['best_model']} <<<")
    print("=" * 65)

    print("Generating Recursive 24-Hour Electricity Demand Forecast...")
    forecast_df = forecast_next_24(eval_res["model"], series)
    print(f"Forecast generated: {len(forecast_df)} rows")
    print("Forecast Table:")
    print(forecast_df.to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main()
