# WATTWISE: Project Architecture & Execution Plan

> **Tagline:** *Predict Demand. Understand Patterns. Stay Ahead.*  
> **Context:** College mini-project, high reliability, simple tabular & time-series architecture, due tomorrow.

---

## 1. System Architecture

WATTWISE is built as a modular Python application with a Streamlit interface. It avoids unnecessary complexity (no React, no external databases, no heavy neural networks/LSTM, no microservices) and focuses on fast, reliable, reproducible electricity demand forecasting.

```
WATTWISE/
├── app.py              # Streamlit dashboard interface
├── requirements.txt    # Production & development dependencies
├── README.md           # Project documentation and quickstart
├── PLAN.md             # Architecture, data flow, leakage rules, and phase checklist
├── pytest.ini          # Pytest configuration
├── data/               # Raw and processed electricity consumption datasets
│   └── README.md
├── src/                # Core business and ML logic
│   ├── __init__.py     # Package initialization
│   ├── data_loader.py  # Ingestion, cleaning, resampling to hourly kW
│   ├── features.py     # Calendar, lag, and rolling window feature extraction
│   ├── models.py       # Temporal splitting, baseline & ML models, evaluation
│   └── insights.py     # Peak detection, anomalies, summary analytics
└── tests/              # Automated unit tests
    └── test_structure.py
```

### Module Responsibilities

| Module | Primary Responsibility |
|---|---|
| `src/data_loader.py` | Parse timestamps, handle nulls/duplicates, aggregate/resample to regular hourly kW demand. |
| `src/features.py` | Generate calendar indicators, lag variables, and rolling statistical aggregations. |
| `src/models.py` | Perform temporal splits, train baseline and ML regressors (e.g., Ridge, Random Forest), evaluate metrics (MAE, RMSE, MAPE, $R^2$), and predict. |
| `src/insights.py` | Compute daily/weekly peak load, base load, detect spikes/anomalies, and formulate plain-English takeaways. |
| `app.py` | Interactive dashboard for data upload/selection, model evaluation, forecast visualization, and operational insights. |

---

## 2. End-to-End Data Flow

The data pipeline follows a strict, unidirectional flow:

```
[ Raw Time-Series Data ]
          │ (CSV / Ingestion)
          ▼
[ Hourly kW Resampling & Cleaning ]  (src/data_loader.py)
          │ (Uniform datetime index, missing value imputation)
          ▼
[ Feature Engineering ]              (src/features.py)
          │ (Calendar: hour, day, month; Lags: t-1, t-24, t-168; Rolling: mean, std)
          ▼
[ Temporal Train / Test Split ]      (src/models.py)
          │ (Chronological cutoff: 80% train, 20% test — NO SHUFFLE)
          ▼
[ Model Training & Baseline ]        (src/models.py)
          │ (Naive/Ridge baseline vs. Random Forest/Gradient Boosting)
          ▼
[ Evaluation & Forecasting ]         (src/models.py)
          │ (Compute MAE, RMSE, MAPE, R²; generate future horizon demand)
          ▼
[ Operational Insights Engine ]      (src/insights.py)
          │ (Peak hours, anomaly alerts, cost implications)
          ▼
[ Streamlit Interactive Dashboard ]  (app.py)
          (Plotly charts, KPI metrics cards, downloadables)
```

---

## 3. Strict Leakage Prevention Rules

Time-series forecasting models are exceptionally vulnerable to subtle data leakage that yields artificially high validation scores but fails in practice. WATTWISE enforces the following five non-negotiable rules:

1. **Chronological Train/Test Split Only:**
   - Shuffling or standard $k$-fold cross-validation is strictly prohibited.
   - The test set must always represent a continuous block of time strictly after the training period.
2. **Lag Features Must Respect Prediction Horizon:**
   - For an $H$-step ahead forecast, all lag features must be $\ge H$ steps in the past, or recursive/direct multi-step forecasting must be properly aligned.
   - In 1-hour ahead evaluation, lag $t-k$ must have $k \ge 1$.
3. **Rolling Features Must Exclude Current Time Step:**
   - Any rolling window (e.g., 24-hour moving average) must be shifted by at least 1 step (`.shift(1).rolling(...)`) so $y_t$ is never included in the statistics used to predict $y_t$.
4. **Scaler / Imputer Fitting Strictness:**
   - Any transformers, scalers, or imputers must be fitted **exclusively on the training dataset** and subsequently applied to test/future sets.
5. **No Lookahead in Exogenous Features:**
   - Only strictly deterministic future variables (such as hour of day, day of week, holidays) may be known ahead of time. No target-derived future variables can enter feature sets.

---

## 4. Phase Checklist

### Phase 0: Scaffolding & Setup (Done)
- [x] Inspect workspace and verify Python environment (Python 3.12+).
- [x] Create project directory structure (`src/`, `data/`, `tests/`).
- [x] Create `requirements.txt` (`streamlit`, `pandas`, `numpy`, `scikit-learn`, `plotly`, `pytest`).
- [x] Initialize virtual environment (`.venv`) and install dependencies.
- [x] Verify package imports via command line.
- [x] Create stubbed `src/` modules with complete docstrings.
- [x] Build minimal Streamlit `app.py`.
- [x] Verify Streamlit launches in headless mode.

### Phase 1: Data Ingestion & Preprocessing (Completed)
- [x] Implement `src/data_loader.py` functions:
  - Load sample electricity demand datasets (UCI household power & compatible CSVs).
  - Robust timestamp parsing and sorting.
  - Hourly resampling (`1h`) and power metric aggregation to hourly mean kW.
  - Time-based interpolation limited to 3 consecutive hours for small gaps.
  - Synthetic benchmark generator with realistic diurnal/weekly/annual patterns.
  - Graceful fallback with clear dataset labelling (`get_data`).
  - Data summary and integrity validation.
- [x] Add unit tests for `data_loader.py` covering '?' handling, duplicates, hourly mean, monotonicity, missing file, and synthetic labelling.
- [x] Provide graceful UCI dataset download helper script (`scripts/download_uci.py`).

### Phase 2: Feature Engineering & Modeling (Completed)
- [x] Implement `src/features.py`:
  - `build_features(series)`: lag features (`lag_1`, `lag_2`, `lag_3`, `lag_24`, `lag_168`), shifted rolling means (`roll_mean_24`, `roll_mean_168`), and calendar features (`hour`, `dayofweek`).
  - `make_row(history_series, timestamp)`: identical feature generation for one future timestamp strictly using past history.
- [x] Implement `src/models.py`:
  - `chrono_split(df, test_frac=0.2)`: chronological train/test splitting with no overlap or shuffling.
  - `seasonal_naive(series, index)`: 24h persistence baseline.
  - `train_rf(X_train, y_train)`: RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42).
  - `metrics(y_true, y_pred)`: MAE, RMSE, and conditional MAPE (actual >= 0.1 kW).
  - `evaluate(series)`: identical test set evaluation comparing Seasonal Naive vs. Random Forest with dynamic best model selection by lowest RMSE.
  - `forecast_next_24(model, series)`: recursive 24-step forecasting starting strictly 1h after the latest observation.
- [x] Add comprehensive tests in `tests/test_models.py` verifying no leakage, chronological split, 24-step recursive forecast shape, known metrics, and seasonal naive lag equivalence.
- [x] Verified on real UCI dataset (Random Forest achieved MAE 0.3305 kW, RMSE 0.4827 kW vs. Baseline MAE 0.5361 kW, RMSE 0.7905 kW).

### Phase 3: Dashboard Integration & Analytics Insights (Completed)
- [x] Implement `src/insights.py`:
  - `peak_summary(forecast_df, threshold)`: flags exceeding hours, identifies peak time and load, and outputs status.
  - `hourly_profile(series)`: 24-hour diurnal average load profile.
  - `weekday_profile(series)`: weekly Monday–Sunday average load dynamics.
  - `top_hours(series, n=3)`: ranked top peak load hours.
  - `data_driven_observations(series)`: 3 data-driven observations computed dynamically from time-series observations.
- [x] Build Streamlit dashboard in `app.py`:
  - Visual Theme: dark charcoal (`#0F1412`), teal (`#14B8A6`), lime (`#A3E635`), rounded cards, `.streamlit/config.toml`.
  - Caching: `@st.cache_data` and `@st.cache_resource` for zero retraining on UI interactions or page navigation.
  - Sidebar Navigation: Overview, Forecast, PeakWatch, Model Arena, Pattern Lens.
  - Synthetic Demo warning banner when real dataset is not detected.
  - Unit clarity banner: hourly average power (kW) vs electrical energy (kWh).
- [x] Comprehensive test coverage in `tests/test_insights.py` and `tests/test_app.py` via `streamlit.testing.v1.AppTest` verifying zero exceptions across all 5 navigation tabs.

### Phase 4: End-to-End Verification & Integration (Completed)
- [x] Full Pytest suite verification: 24 tests passing across data loader, models, feature leakage, insights, and Streamlit AppTest.
- [x] Three data mode tests:
  - (a) Real UCI data in `data/` (verified 34,190 hourly kW observations).
  - (b) Compatible custom CSV upload with Date+Time or single datetime column.
  - (c) Synthetic Demo Mode fallback with prominent warning banner.
- [x] Failure cases verification: empty file (0 bytes), missing required columns, and malformed CSV syntax handled gracefully via `st.error` without application crash.
- [x] End-to-end verification:
  - Forecast timestamps start strictly 1h after the last observation.
  - UI metrics equal backend evaluation metrics from `models.evaluate`.
  - Best-model label matches lower RMSE.
  - PeakWatch peak equals forecast maximum.
  - Threshold slider does not retrigger model training (sub-millisecond cache execution).
- [x] Zero-leakage verification: features at $t$ strictly invariant to alterations at $\ge t$.
- [x] Performance optimization: hourly series cached to `data/hourly.parquet`, reducing reload latency from ~3.1s to < 0.1s.

### Phase 5: Final Documentation & Presentation Polish (Completed)
- [x] Finalized `requirements.txt` with exact pinned versions from `pip freeze` for only the 7 packages directly used.
- [x] Comprehensive `README.md` covering overview, tagline, 5 features, project tree, UCI dataset specifics (DOI 10.24432/C58K54), Windows setup, test execution, methodology, limitations, and future extensions.
- [x] Created `DEMO.md`: complete 5-minute timed presentation script, pre-demo checklist, and viva defense Q&A cheat sheet.
- [x] Created `TECHNICAL_NOTES.md`: plain-language technical architecture and module-by-module breakdown.
- [x] Fresh-environment validation: created `.venv_fresh`, installed pinned dependencies, verified all 24 pytest tests pass, and launched Streamlit cleanly without errors. Cleaned up temporary environment.

---
**Project Status:** 100% Complete, Validated, and Submission-Ready.
