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

### Phase 2: Feature Engineering & Modeling (Next)
- [ ] Implement `src/features.py` functions:
  - Calendar attributes (`hour`, `dayofweek`, `month`, `is_weekend`).
  - Cyclical encoding (`sin`/`cos` transforms for hour and day).
  - Historical lags (`t-1`, `t-2`, `t-24`, `t-168`).
  - Shifted rolling statistics (`rolling_mean_24h`, `rolling_std_24h`).
- [ ] Ensure non-leakage verification tests in `tests/test_features.py`.

### Phase 3: Modeling & Evaluation
- [ ] Implement `src/models.py` functions:
  - Chronological split utility (`train_test_split_temporal`).
  - Baseline model (Persistence / Linear Ridge).
  - ML model (Random Forest Regressor / Gradient Boosting).
  - Evaluation metric suite (`MAE`, `RMSE`, `MAPE`, $R^2$).
  - Future horizon forecast inference.
- [ ] Add unit tests for model training and metric computation.

### Phase 4: Insights Engine
- [ ] Implement `src/insights.py` functions:
  - Peak demand hour detection and weekday vs. weekend profile comparison.
  - Anomaly and consumption spike detection using moving IQR / Z-score.
  - Plain-English actionable recommendations for peak shaving and load shifting.

### Phase 5: Streamlit Dashboard UI & Delivery
- [ ] Implement `app.py` UI sections:
  - Sidebar: Data source selector, forecast horizon slider, model selector.
  - Tab 1: Historical Consumption & Pattern Analysis (Plotly interactive charts).
  - Tab 2: Forecast vs. Actual & Evaluation Metrics (KPI cards, comparison plots).
  - Tab 3: Operational Insights & Anomaly Alerts.
- [ ] End-to-end verification and demonstration readiness for presentation.
