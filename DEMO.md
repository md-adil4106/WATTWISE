# WATTWISE: 5-Minute Live Demo & Viva Presentation Guide

> **Project Tagline:** *Predict Demand. Understand Patterns. Stay Ahead.*  
> **Target Audience:** College mini-project examiners, evaluators, and engineering peers.

---

## 📋 Pre-Demo Checklist (Complete 10 Minutes Before Presentation)

- [ ] **1. Working Directory:** Open terminal in project root (`c:\Users\Mohommed Adil\Desktop\WATTWISE`).
- [ ] **2. Virtual Environment:** Ensure `.venv` is activated:
  ```powershell
  .\.venv\Scripts\activate
  ```
- [ ] **3. Automated Test Verification:** Run test suite to verify 100% green status:
  ```powershell
  pytest -v
  ```
  *(Confirms all 24 unit, model, leakage, and UI integration tests pass).*
- [ ] **4. Dataset Cache Warmup:** Check that `data/hourly.parquet` exists for sub-second startup (< 0.1s load time).
- [ ] **5. Launch Dashboard:**
  ```powershell
  streamlit run app.py
  ```
- [ ] **6. Browser Setup:** Open `http://localhost:8501`, set browser zoom to 100% (or 110% for projector clarity), expand the window, and verify sidebar navigation.

---

## ⏱️ 5-Minute Presentation Script

### Minute 1:00 — Project Vision & Motivation
- **Speaker:**  
  *"Good morning evaluators. Today I am presenting **WATTWISE**, an electricity demand forecasting and time-series analytics dashboard built to address a fundamental challenge in electrical power engineering: grid capacity management and load predictability."*
- **Visual:** Point to top header with logo `⚡ WATTWISE` and tagline: *"Predict Demand. Understand Patterns. Stay Ahead."*
- **Key Talking Points:**
  - In power grids, supply must match instantaneous demand in real time. Sudden demand spikes can trip distribution transformers or trigger costly peak-peaking generation.
  - The project utilizes real smart-meter telemetry from the **UCI Machine Learning Repository** comprising over 2 million raw measurements gathered over 47 months.
  - Highlight the **Unit Notice**: *"All measurements represent hourly average power in kilowatts (kW), not total accumulated energy in kilowatt-hours (kWh)."*

---

### Minute 2:00 — Overview & 24-Hour Horizon Forecasting
- **Action:** Remain on the **Overview** page.
- **Visual:** Highlight the 4 KPI cards and the central 7-day actuals + 24-hour forecast curve.
- **Speaker:**  
  *"On the Overview screen, an energy manager immediately sees four critical operational signals: the latest recorded load with its exact timestamp, the predicted 24-hour mean demand, the predicted peak load timing, and model validation accuracy."*
- **Action:** Hover over the interactive Plotly chart.
- **Key Talking Points:**
  - Notice the solid teal line representing the past 7 days of actual consumption.
  - Observe how the dashed lime curve seamlessly forecasts the upcoming 24 hours.
  - Point to the **Data-Driven Insight card** at the bottom, explaining how peak demand is automatically contextualized against the daily baseline.

---

### Minute 3:00 — PeakWatch: Threshold Alerts & Operational Safeguards
- **Action:** Click **PeakWatch** in the sidebar.
- **Speaker:**  
  *"One of the central operational capabilities of WATTWISE is **PeakWatch**. Facility managers and grid operators need advance notice when load threatens to exceed contractual or equipment thresholds."*
- **Action:** Drag the threshold slider down slightly below the predicted peak (e.g., from the default 90th percentile of ~2.2 kW down to ~1.8 kW).
- **Visual:** Show how the chart dynamically highlights violating hours with red markers and changes status from green to an active red `⚠️ Peak Alert`.
- **Key Talking Points:**
  - Emphasize performance: moving the slider does **NOT** retrain the model; thanks to `@st.cache_resource`, inference is decoupled and updates in under 1 millisecond.
  - Point to the statutory disclaimer at the bottom: predictions are statistical guidelines and operational estimates, not deterministic guarantees.

---

### Minute 4:00 — Model Arena: Baseline vs. Machine Learning Benchmark
- **Action:** Click **Model Arena** in the sidebar.
- **Speaker:**  
  *"In machine learning, it is trivial to report high accuracy if a model is evaluated against a random split or without a strong benchmark. In WATTWISE, we enforce strict chronological splitting with zero lookahead leakage."*
- **Visual:** Point to the **Evaluation Metrics Table** and the Champion Model banner.
- **Key Talking Points:**
  - We compare a 24-hour persistence baseline (`Seasonal Naive`) against our primary `Random Forest Regressor` (200 trees, max depth 12).
  - Both models are evaluated on the exact same test timestamps (the final 20% chronological holdout block).
  - Walk through the numbers: Random Forest achieves **0.3305 kW MAE** and **0.4827 kW RMSE**, delivering a **38% error reduction** over the baseline.
  - Point out that the champion model is dynamically determined from lowest RMSE, never hardcoded.

---

### Minute 5:00 — Pattern Lens, Methodology & Examiner Conclusion
- **Action:** Click **Pattern Lens** in the sidebar.
- **Speaker:**  
  *"Finally, the Pattern Lens module breaks down the behavioral energy signatures embedded within the data."*
- **Visual:** Show the diurnal hourly bar chart and the weekly rhythm chart.
- **Key Talking Points:**
  - Daily cycle shows an overnight trough around 04:00 (averaging 0.44 kW base load), a morning ramp at 07:00, and an evening peak at 20:00 (averaging ~1.90 kW).
  - Weekly rhythm reveals weekends average ~19% higher demand than weekdays due to residential occupancy.
  - Explain the 3 data-driven observations generated dynamically from the series.
- **Conclusion:**  
  *"In conclusion, WATTWISE demonstrates a clean, leak-free, modular architecture designed for high reliability without unnecessary framework overhead. Thank you, and I welcome your questions."*

---

## 🎯 Viva Defense / Examiner Q&A Cheat Sheet

| Question | Recommended Answer |
|---|---|
| **Why did you use Random Forest instead of LSTM or Deep Learning?** | For tabular time-series with lag and rolling features, tree-based ensembles (Random Forest / Gradient Boosting) train in seconds, avoid catastrophic overfitting on single-household data, require no GPU hardware, and provide reliable, deterministic predictions without complex neural hyperparameter tuning. |
| **How did you prevent data leakage?** | Five strict rules: (1) Chronological train/test split only (no random shuffle). (2) Lags strictly use $t-k$ ($k \ge 1$). (3) Rolling statistics are calculated on `series.shift(1)` so $y_t$ is excluded. (4) Shared `make_row` function enforces that predictions strictly rely on history $< t$. (5) Automated tests verify features at $t$ are 100% invariant when future data is altered. |
| **Why is power measured in kW rather than kWh?** | Kilowatts (kW) represents the instantaneous rate of power demand. Electrical equipment, lines, and transformers are sized based on peak power capacity (kW). Kilowatt-hours (kWh) measures cumulative energy consumed over time. WATTWISE forecasts load demand rate (kW) to prevent equipment overload. |
| **How does the recursive 24-hour forecast work?** | In step 1, the model uses actual historical lags to predict $\hat{y}_{t+1}$. For step 2, $\hat{y}_{t+1}$ is appended to the series to act as lag 1 for $\hat{y}_{t+2}$. This recurses 24 times to forecast the full 24-hour horizon. |
| **What happens if someone runs the app without the dataset?** | WATTWISE detects missing files gracefully and automatically transitions to **SYNTHETIC DEMO** mode (generating 365 days of realistic diurnal/weekly patterns) with a persistent warning banner, ensuring the demo never crashes. |
