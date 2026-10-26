# WATTWISE ⚡

> **Predict Demand. Understand Patterns. Stay Ahead.**

An electricity-demand forecasting dashboard built for high reliability, simplicity, and actionable insights. Designed as a college mini-project to demonstrate end-to-end time-series machine learning, from raw consumption data to interactive visual forecasts and peak demand analytics.

---

## Key Features

- **End-to-End Pipeline:** Raw data ingestion $\rightarrow$ Hourly kW cleaning $\rightarrow$ Feature engineering $\rightarrow$ Temporal split $\rightarrow$ ML forecasting $\rightarrow$ Interactive dashboard.
- **Zero Data Leakage:** Strict chronological splitting and shifted rolling statistics to prevent lookahead bias.
- **Interpretable Models:** Compares naive baselines against tabular machine learning models (Random Forest / Gradient Boosting).
- **Operational Insights:** Automated peak load identification, anomaly detection, and energy management recommendations.
- **Streamlit & Plotly Interface:** Clean, lightweight web interface without the overhead of external databases or complex frontend frameworks.

---

## Project Structure

```
WATTWISE/
├── app.py              # Streamlit dashboard interface
├── requirements.txt    # Project dependencies
├── PLAN.md             # Architecture, data flow, leakage rules & checklist
├── README.md           # Project documentation
├── pytest.ini          # Pytest configuration
├── data/               # Raw and processed electricity consumption datasets
│   └── README.md
├── src/                # Core modular Python packages
│   ├── __init__.py
│   ├── data_loader.py  # Data loading, timestamp cleaning, hourly resampling
│   ├── features.py     # Calendar, lag, and rolling features
│   ├── models.py       # Temporal splitting, training, evaluation, forecasting
│   └── insights.py     # Peak hours, anomalies, operational takeaways
└── tests/              # Test suite
    ├── __init__.py
    └── test_structure.py
```

---

## Quickstart

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.12)

### 2. Setup Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Or on macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Dataset Setup
Download the [UCI Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) (DOI: `10.24432/C58K54`, Repo ID: `235`) and place the extracted file at:
```
data/household_power_consumption.txt
```
*(Note: Raw dataset files are excluded from Git via `.gitignore` to prevent tracking large files).*

Alternatively, an optional helper script is provided to attempt downloading the UCI archive automatically:
```bash
python scripts/download_uci.py
```
If network connectivity is unavailable or the download fails, the script fails gracefully and WATTWISE automatically runs in **SYNTHETIC DEMO** mode without crashing.

### 5. Run the Dashboard
```bash
streamlit run app.py
```

### 6. Run Tests
```bash
pytest
```
