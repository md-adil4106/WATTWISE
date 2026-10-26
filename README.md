# WATTWISE ⚡

> **Predict Demand. Understand Patterns. Stay Ahead.**

WATTWISE is an end-to-end electricity consumption forecasting and time-series analytics dashboard. Designed as a college mini-project, it bridges raw 1-minute smart-meter readings with machine learning to deliver recursive 24-hour demand predictions, automated capacity peak alerts, and deep behavioral energy insights—all within an interactive, dark-themed Streamlit interface.

---

## 🚀 Key Dashboard Features

1. **Overview:**
   - Real-time operational KPI metric cards: latest recorded hourly kW, predicted 24-hour mean kW, predicted peak demand and timestamp, and model error metrics.
   - 7-day historical load profile seamlessly bridged with the 24-hour predictive horizon.
   - Plain-English operational takeaway card.

2. **Forecast (Forecast Horizon):**
   - High-resolution interactive Plotly line chart distinguishing historical actuals (solid teal `#14B8A6`) from recursive 24-step predictions (dashed lime `#A3E635`).
   - Granular tabular breakdown of step-by-step forecasted kW demand.
   - Interactive Timestamp Inspector slider to examine target hours in detail.

3. **PeakWatch (Capacity & Threshold Monitoring):**
   - Dynamic threshold slider defaulting to the 90th percentile of historical consumption.
   - Visual violation detection with highlighted markers on the forecast curve.
   - Status indicators (`No threshold violations predicted` vs. active peak alert with excess magnitude).
   - Clear operational disclaimer: predictions are statistical guidelines, not guaranteed grid events.

4. **Model Arena (Benchmark & Validation):**
   - Head-to-head empirical comparison between a 24-hour Persistence Baseline (`Seasonal Naive`) and a 200-tree `Random Forest Regressor`.
   - Comprehensive error metrics table (MAE, RMSE, and conditional MAPE).
   - Dynamic champion model declaration based on lowest RMSE.
   - Interactive test set visualizer (last 3, 7, or 14 days) and comparative error bar charts.

5. **Pattern Lens (Behavioral Dynamics):**
   - 24-hour diurnal load signature highlighting morning ramps and evening peak windows.
   - Weekly load rhythm comparison (Monday through Sunday).
   - Ranked top peak-demand hours of the day.
   - 3 dynamic data-driven observations computed directly from the active dataset.

---

## 📁 Project Structure

```
WATTWISE/
├── app.py                   # Streamlit interactive dashboard application
├── requirements.txt         # Pinned production & development dependencies
├── PLAN.md                  # Comprehensive execution plan, architecture & leakage rules
├── README.md                # Project documentation and quickstart guide
├── DEMO.md                  # 5-minute presentation script and pre-demo checklist
├── TECHNICAL_NOTES.md       # Module-by-module architectural breakdown
├── pytest.ini               # Pytest configuration
├── .streamlit/
│   └── config.toml          # Dark charcoal theme configuration
├── data/                    # Dataset directory (raw data excluded from Git)
│   ├── README.md            # Dataset specification and acquisition guide
│   ├── household_power_consumption.txt  # Real UCI dataset (local)
│   └── hourly.parquet       # Processed hourly cache (< 0.1s startup)
├── src/                     # Core modular Python packages
│   ├── __init__.py          # Package initialization
│   ├── data_loader.py       # Ingestion, cleaning, hourly resampling, synthetic demo
│   ├── features.py          # Lags, shifted rolling means, calendar indicators
│   ├── models.py            # Chronological split, baseline, Random Forest, 24h forecasting
│   └── insights.py          # Peak detection, profile analysis, data-driven takeaways
├── scripts/                 # Standalone automation & verification utilities
│   ├── download_uci.py      # Optional helper to download UCI dataset archive
│   ├── verify_phase2.py     # Verification runner for modeling and forecasting
│   └── verify_phase4.py     # End-to-end integration & leakage verification
└── tests/                   # Automated pytest suite (24 unit & UI tests)
    ├── __init__.py
    ├── test_app.py          # Streamlit UI tests via streamlit.testing AppTest
    ├── test_data.py         # Data loading, resampling, gap interpolation tests
    ├── test_features.py     # Feature engineering tests
    ├── test_insights.py     # Peak detection and profiling tests
    ├── test_models.py       # Leakage tests, chronological split, 24h forecast tests
    └── test_structure.py    # Architectural and callable integrity tests
```

---

## 📊 Dataset: UCI Individual Household Electric Power Consumption

- **Dataset Name:** Individual Household Electric Power Consumption
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) (Repository ID: `235`)
- **DOI:** [`10.24432/C58K54`](https://doi.org/10.24432/C58K54)
- **Observations:** 2,075,259 measurements gathered over 47 months (December 2006 – November 2010).
- **Sampling Frequency:** 1-minute intervals.
- **Local Placement:**
  Place the extracted file directly into the `data/` folder:
  ```
  data/household_power_consumption.txt
  ```
  *(Note: Raw data files are excluded from Git via `.gitignore` to prevent tracking ~133MB files).*

- **Automatic Download Script:**
  An optional helper script is provided to download the archive automatically:
  ```bash
  python scripts/download_uci.py
  ```
  If network connectivity is unavailable or the download fails, the script fails gracefully.

- **Synthetic Demo Fallback:**
  If the real UCI dataset is absent, WATTWISE automatically falls back to **SYNTHETIC DEMO** mode (generating 365 days of realistic diurnal, weekly, and seasonal hourly demand). Synthetic demo mode is clearly flagged with a prominent warning banner on every page and is never falsely labeled as UCI data.

---

## 💻 Windows Setup & Running Instructions

### 1. Prerequisites
- Python 3.10 to 3.12 (Tested on Python 3.12.10 on Windows 11).
- Git.

### 2. Create Virtual Environment
Open PowerShell in the project directory:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Run Automated Tests
```powershell
pytest
```
Or with verbose output:
```powershell
pytest -v
```

---

## 🔬 Methodology

1. **Hourly Resampling & Cleaning:**
   - Raw 1-minute measurements contain missing values (`?`), negative outliers, and occasional duplicate timestamps.
   - Timestamps are parsed to a continuous DatetimeIndex and deduplicated.
   - Readings are resampled to hourly arithmetic means ($\text{kW} = \frac{1}{60} \sum_{m=1}^{60} P_m$).
   - Small gaps ($\le 3$ consecutive hours) are imputed using forward time-based interpolation; larger gaps remain unobserved and are dropped.
   - Resulting series is named `'kw'` (hourly average power in kW, not energy in kWh).

2. **Feature Engineering with Zero Leakage:**
   - **Historical Lags:** $t-1$, $t-2$, $t-3$, $t-24$ (yesterday same hour), and $t-168$ (last week same hour).
   - **Shifted Rolling Statistics:** 24-hour and 168-hour moving averages computed strictly on `series.shift(1)` so the current hour $y_t$ is never included in its own input feature set.
   - **Calendar Deterministics:** Hour of day (0–23) and day of week (0–6).

3. **Strict Chronological Evaluation:**
   - Standard random shuffling or k-fold cross-validation causes catastrophic lookahead bias in time series.
   - WATTWISE splits the dataset chronologically: initial 80% partition for training, final 20% continuous block for test evaluation (`train.index.max() < test.index.min()`).

4. **Forecasting Models:**
   - **Baseline (`Seasonal Naive`):** Predicts $\hat{y}_t = y_{t-24\text{h}}$ (daily persistence).
   - **Primary Model (`Random Forest Regressor`):** Ensemble of 200 trees (`max_depth=12`, `random_state=42`) trained on lag and rolling features.

5. **Recursive 24-Hour Forecasting:**
   - Multi-step horizon is generated recursively for 24 steps ($t+1$ to $t+24$).
   - Step 1 builds features from actual history and predicts $\hat{y}_{t+1}$.
   - $\hat{y}_{t+1}$ is appended to the sequence to supply the lag inputs for $\hat{y}_{t+2}$, repeating recursively for all 24 hours.

6. **Evaluation Metrics:**
   - **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{N} \sum |y_t - \hat{y}_t|$ (average kW magnitude error).
   - **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\frac{1}{N} \sum (y_t - \hat{y}_t)^2}$ (penalizes large peak misses).
   - **Conditional MAPE:** Mean absolute percentage error computed strictly on actuals $\ge 0.1\text{ kW}$ to avoid division-by-zero distortion during low overnight baseloads.

---

## ⚠️ Limitations

1. **Single Household Scope:** The UCI dataset captures energy demand from a single residential home in Sceaux, France; individual household profiles exhibit higher stochastic variance than aggregated municipal grid loads.
2. **Absence of Exogenous Weather Regressors:** Heating and cooling loads correlate strongly with ambient temperature, humidity, and solar irradiance. The current models rely strictly on historical demand and calendar proxies without live meteorological feeds.
3. **Recursive Error Propagation:** Multi-step recursive forecasting carries predictions forward as lag inputs for subsequent steps; errors in earlier hours ($t+1$ to $t+4$) can compound into later horizon steps ($t+20$ to $t+24$).
4. **Statistical Estimates vs. Physical Guarantees:** Forecasts are statistical expected values and operational guidelines, not hard deterministic guarantees.

---

## 🔮 Future Work

1. **SARIMAX Integration:** Incorporate Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors (SARIMAX) to model linear seasonal differencing alongside exogenous weather inputs.
2. **Gradient Boosting (XGBoost / LightGBM):** Benchmark tree-based gradient boosting ensembles for faster training, native handling of missing values, and potential incremental accuracy gains over Random Forest.
3. **Deep Learning Sequence Models (LSTM / Temporal Fusion Transformers):** Implement Long Short-Term Memory (LSTM) recurrent networks and Temporal Fusion Transformers (TFT) with quantile loss for probabilistic prediction intervals.
4. **Automated Weather API Ingestion:** Integrate Open-Meteo or NOAA APIs to supply temperature, wind chill, and humidity as dynamic exogenous features.
