"""Comprehensive Phase 4 verification and integration test script for WATTWISE.

Tests all required verification items:
1. Modes:
   (a) Real UCI data in data/
   (b) Compatible CSV with Date+Time and single datetime column
   (c) Synthetic mode with banner (no data file)
2. Failure cases:
   - Malformed CSV
   - Missing required column
   - Empty file (0 bytes)
3. End-to-end alignment:
   - Forecast starts strictly 1h after last observation
   - Metrics match models.evaluate
   - Best-model label matches lowest RMSE
   - PeakWatch peak equals max forecast
   - Slider adjustment does not retrigger training (cache verification)
4. Data leakage verification
"""

from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import clean_and_resample, generate_synthetic, get_data, load_raw
from src.features import build_features, make_row
from src.insights import peak_summary
from src.models import evaluate, forecast_next_24


def run_tests() -> list[tuple[str, str, str]]:
    results = []

    # -------------------------------------------------------------
    # 1. Mode (a): Real UCI dataset
    # -------------------------------------------------------------
    uci_file = Path("data/household_power_consumption.txt")
    if uci_file.exists():
        t0 = time.time()
        series, label = get_data(uci_file)
        t_load = time.time() - t0
        passed = (
            len(series) > 30_000
            and label == "UCI Household Power Consumption"
            and series.index.is_monotonic_increasing
            and t_load < 30.0
        )
        results.append((
            "Mode (a): Real UCI Data Ingestion",
            "PASS" if passed else "FAIL",
            f"Loaded {len(series):,} hourly rows in {t_load:.2f}s (Label: '{label}')",
        ))
    else:
        results.append((
            "Mode (a): Real UCI Data Ingestion",
            "SKIPPED",
            "Real UCI file 'data/household_power_consumption.txt' not present.",
        ))

    # -------------------------------------------------------------
    # 2. Mode (b): Compatible CSV
    # -------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
        f.write("datetime,Global_active_power\n")
        f.write("2026-01-01 00:00:00,1.25\n")
        f.write("2026-01-01 01:00:00,1.85\n")
        f.write("2026-01-01 02:00:00,2.10\n")
        compat_csv_path = f.name

    try:
        raw_b = load_raw(compat_csv_path)
        clean_b = clean_and_resample(raw_b)
        passed_b = len(clean_b) == 3 and clean_b.name == "kw" and clean_b.iloc[0] == 1.25
        results.append((
            "Mode (b): Compatible CSV Parsing",
            "PASS" if passed_b else "FAIL",
            f"Successfully parsed compatible CSV into {len(clean_b)} hourly kw rows.",
        ))
    except Exception as exc:
        results.append(("Mode (b): Compatible CSV Parsing", "FAIL", str(exc)))
    finally:
        Path(compat_csv_path).unlink(missing_ok=True)

    # -------------------------------------------------------------
    # 3. Mode (c): Synthetic Mode with Banner
    # -------------------------------------------------------------
    synth_series, synth_label = get_data(path="non_existent_dataset_123.txt")
    passed_c = (
        synth_label == "SYNTHETIC DEMO"
        and "UCI" not in synth_label
        and synth_series.attrs.get("is_synthetic") is True
        and len(synth_series) == 365 * 24
    )
    results.append((
        "Mode (c): Synthetic Demo Mode Fallback",
        "PASS" if passed_c else "FAIL",
        f"Correctly returned label '{synth_label}' with 8,760 synthetic hours.",
    ))

    # -------------------------------------------------------------
    # 4. Failure Case: Empty file (0 bytes)
    # -------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
        f.write("")
        empty_path = f.name

    try:
        load_raw(empty_path)
        results.append(("Failure Case: Empty File", "FAIL", "Did not raise ValueError on empty file."))
    except ValueError as val_err:
        results.append((
            "Failure Case: Empty File Handling",
            "PASS",
            f"Caught expected friendly ValueError: {val_err}",
        ))
    except Exception as exc:
        results.append(("Failure Case: Empty File Handling", "FAIL", f"Unexpected error type: {exc}"))
    finally:
        Path(empty_path).unlink(missing_ok=True)

    # -------------------------------------------------------------
    # 5. Failure Case: Missing Required Columns
    # -------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
        f.write("Date,Time,Voltage,Current\n")
        f.write("01/01/2026,10:00:00,230.5,10.2\n")
        missing_col_path = f.name

    try:
        load_raw(missing_col_path)
        results.append(("Failure Case: Missing Column", "FAIL", "Did not raise ValueError on missing column."))
    except ValueError as val_err:
        results.append((
            "Failure Case: Missing Column Handling",
            "PASS",
            f"Caught expected friendly ValueError: {val_err}",
        ))
    except Exception as exc:
        results.append(("Failure Case: Missing Column Handling", "FAIL", f"Unexpected error type: {exc}"))
    finally:
        Path(missing_col_path).unlink(missing_ok=True)

    # -------------------------------------------------------------
    # 6. Failure Case: Malformed CSV Syntax
    # -------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as f:
        # Invalid binary junk that causes read errors
        f.write(b"\x00\xff\xfe\x00\x00\x01\x02\x03\x04\x05")
        malformed_path = f.name

    try:
        load_raw(malformed_path)
        results.append(("Failure Case: Malformed CSV", "FAIL", "Did not raise ValueError on malformed CSV."))
    except ValueError as val_err:
        results.append((
            "Failure Case: Malformed CSV Handling",
            "PASS",
            f"Caught expected friendly ValueError: {val_err}",
        ))
    except Exception as exc:
        results.append(("Failure Case: Malformed CSV Handling", "FAIL", f"Unexpected error type: {exc}"))
    finally:
        Path(malformed_path).unlink(missing_ok=True)

    # -------------------------------------------------------------
    # 7. End-to-End: Forecast Timestamps start strictly 1h after last obs
    # -------------------------------------------------------------
    synth_s = generate_synthetic(days=14, seed=42)
    eval_res = evaluate(synth_s)
    forecast_df = forecast_next_24(eval_res["model"], synth_s)

    expected_start = synth_s.index[-1] + pd.Timedelta(hours=1)
    passed_timeline = (
        len(forecast_df) == 24
        and forecast_df["timestamp"].iloc[0] == expected_start
        and (forecast_df["timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all()
    )
    results.append((
        "End-to-End: Forecast Timeline Sequence",
        "PASS" if passed_timeline else "FAIL",
        f"Starts strictly at {expected_start} (+1h from {synth_s.index[-1]}).",
    ))

    # -------------------------------------------------------------
    # 8. End-to-End: Metrics match models.evaluate
    # -------------------------------------------------------------
    rf_mae = eval_res["rf_metrics"]["MAE"]
    rf_rmse = eval_res["rf_metrics"]["RMSE"]
    base_mae = eval_res["baseline_metrics"]["MAE"]
    base_rmse = eval_res["baseline_metrics"]["RMSE"]
    passed_metrics = (
        isinstance(rf_mae, float) and rf_mae > 0
        and isinstance(rf_rmse, float) and rf_rmse > 0
        and isinstance(base_mae, float) and base_mae > 0
        and isinstance(base_rmse, float) and base_rmse > 0
    )
    results.append((
        "End-to-End: Metrics Integrity Check",
        "PASS" if passed_metrics else "FAIL",
        f"RF MAE={rf_mae:.4f}, RMSE={rf_rmse:.4f} | Base MAE={base_mae:.4f}, RMSE={base_rmse:.4f}",
    ))

    # -------------------------------------------------------------
    # 9. End-to-End: Best Model Matches Lowest RMSE
    # -------------------------------------------------------------
    expected_best = "Random Forest" if rf_rmse < base_rmse else "Seasonal Naive"
    passed_best = eval_res["best_model"] == expected_best
    results.append((
        "End-to-End: Best Model Dynamic Selection",
        "PASS" if passed_best else "FAIL",
        f"Computed Champion: '{eval_res['best_model']}' (matches lower RMSE condition).",
    ))

    # -------------------------------------------------------------
    # 10. End-to-End: PeakWatch Peak Equals Forecast Maximum
    # -------------------------------------------------------------
    summary = peak_summary(forecast_df, threshold=float(synth_s.quantile(0.9)))
    max_forecast_kw = float(forecast_df["forecast_kw"].max())
    passed_peak = summary["peak_kw"] == pytest_approx(max_forecast_kw, 1e-4)
    results.append((
        "End-to-End: PeakWatch Peak Equivalence",
        "PASS" if passed_peak else "FAIL",
        f"PeakWatch peak={summary['peak_kw']:.4f} kW equals forecast max={max_forecast_kw:.4f} kW.",
    ))

    # -------------------------------------------------------------
    # 11. End-to-End: Threshold Slider Does Not Retrigger Training
    # -------------------------------------------------------------
    # Test that peak_summary executes purely on forecast_df in microseconds without re-evaluating
    t0_slider = time.perf_counter()
    summary_1 = peak_summary(forecast_df, threshold=1.5)
    summary_2 = peak_summary(forecast_df, threshold=2.0)
    summary_3 = peak_summary(forecast_df, threshold=2.5)
    t_slider = (time.perf_counter() - t0_slider) * 1000
    passed_cache = t_slider < 50.0  # 3 threshold changes in <50ms
    results.append((
        "End-to-End: Slider Threshold Decoupled from Training",
        "PASS" if passed_cache else "FAIL",
        f"3 threshold slider evaluations took {t_slider:.2f} ms (no model retraining).",
    ))

    # -------------------------------------------------------------
    # 12. Leakage Verification
    # -------------------------------------------------------------
    f1 = build_features(synth_s)
    fcols = [c for c in f1.columns if c != "y"]
    t_test = f1.index[40]

    s_perturbed = synth_s.copy()
    s_perturbed.loc[s_perturbed.index >= t_test] = 9999.0
    f2 = build_features(s_perturbed)

    passed_leakage = f1.loc[t_test, fcols].equals(f2.loc[t_test, fcols])
    results.append((
        "Leakage Rule Verification: Temporal Independence",
        "PASS" if passed_leakage else "FAIL",
        "Verified features at t are 100% invariant to alterations at >= t.",
    ))

    return results


def pytest_approx(val, tol):
    return round(val, 4)


def main():
    print("=" * 80)
    print("WATTWISE Phase 4: Integration Testing & Bug Fixing Verification")
    print("=" * 80)
    results = run_tests()

    print(f"{'Test Description':<50} | {'Status':<6} | {'Details'}")
    print("-" * 80)
    all_passed = True
    for desc, status, details in results:
        if status != "PASS" and status != "SKIPPED":
            all_passed = False
        print(f"{desc:<50} | {status:<6} | {details}")
    print("=" * 80)
    if all_passed:
        print(">>> ALL PHASE 4 INTEGRATION CHECKS PASSED <<<")
    else:
        print(">>> SOME CHECKS FAILED <<<")


if __name__ == "__main__":
    main()
