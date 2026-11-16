"""WATTWISE CA3 Model Benchmark Script.

Executes and benchmarks all project models under strict chronological 80/20 split:
1. Seasonal Naive Baseline (t-24h)
2. Random Forest Regressor (200 trees, depth 12)
3. XGBoost Regressor (200 trees, depth 6, lr 0.05)
4. LightGBM Regressor (200 trees, depth 6, lr 0.05)
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.data_loader as dl
import src.features as ft
import src.models as models
import xgboost as xgb
import lightgbm as lgb


def main():
    print("=" * 65)
    print(" WATTWISE: CA3 COMPREHENSIVE MODEL BENCHMARK (REAL EVALUATION)")
    print("=" * 65)

    # 1. Load Data
    t0 = time.time()
    series, label = dl.get_data()
    print(f"[*] Dataset: {label}")
    print(f"[*] Total clean hourly records: {len(series)}")

    # 2. Build Features
    df = ft.build_features(series)
    dropped = len(series) - len(df)
    print(f"[*] Features built: {len(df)} rows ({dropped} rows dropped for 7-day lag)")

    # 3. Chronological Split
    train_df, test_df = models.chrono_split(df, test_frac=0.2)
    print(f"[*] Chronological Split: Train={len(train_df)} rows, Test={len(test_df)} rows")
    print(f"    Train Period: {train_df.index[0]} -> {train_df.index[-1]}")
    print(f"    Test Period:  {test_df.index[0]}  -> {test_df.index[-1]}")
    print("-" * 65)

    feature_cols = [c for c in df.columns if c != "y"]
    X_train, y_train = train_df[feature_cols], train_df["y"]
    X_test, y_test = test_df[feature_cols], test_df["y"]

    results = []

    # Model 1: Seasonal Naive Baseline
    print("[1/4] Evaluating Seasonal Naive Baseline (t-24h)...")
    baseline_series = models.seasonal_naive(series, test_df.index)
    m_naive = models.metrics(y_test.values, baseline_series.values)
    results.append({"Model": "Seasonal Naive (t-24)", "RMSE": m_naive["RMSE"], "MAE": m_naive["MAE"], "MAPE (%)": m_naive["MAPE"]})

    # Model 2: Random Forest
    print("[2/4] Training Random Forest Regressor (200 trees, max_depth 12)...")
    t_rf = time.time()
    rf = models.train_rf(X_train, y_train)
    rf_preds = rf.predict(X_test)
    m_rf = models.metrics(y_test.values, rf_preds)
    results.append({"Model": "Random Forest", "RMSE": m_rf["RMSE"], "MAE": m_rf["MAE"], "MAPE (%)": m_rf["MAPE"]})
    print(f"      Trained in {time.time() - t_rf:.2f}s")

    # Model 3: XGBoost
    print("[3/4] Training XGBoost Regressor (200 estimators, max_depth 6)...")
    t_xgb = time.time()
    xgb_mod = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb_mod.fit(X_train, y_train)
    xgb_preds = xgb_mod.predict(X_test)
    m_xgb = models.metrics(y_test.values, xgb_preds)
    results.append({"Model": "XGBoost", "RMSE": m_xgb["RMSE"], "MAE": m_xgb["MAE"], "MAPE (%)": m_xgb["MAPE"]})
    print(f"      Trained in {time.time() - t_xgb:.2f}s")

    # Model 4: LightGBM
    print("[4/4] Training LightGBM Regressor (200 estimators, max_depth 6)...")
    t_lgb = time.time()
    lgb_mod = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
    lgb_mod.fit(X_train, y_train)
    lgb_preds = lgb_mod.predict(X_test)
    m_lgb = models.metrics(y_test.values, lgb_preds)
    results.append({"Model": "LightGBM", "RMSE": m_lgb["RMSE"], "MAE": m_lgb["MAE"], "MAPE (%)": m_lgb["MAPE"]})
    print(f"      Trained in {time.time() - t_lgb:.2f}s")

    print("-" * 65)
    print("\nFINAL CA3 COMPARISON TABLE (EVALUATED ON TEST SET):")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print(f"\n[*] Total execution time: {time.time() - t0:.2f}s\n")
    print("=" * 65)


if __name__ == "__main__":
    main()
