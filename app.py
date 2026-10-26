"""WATTWISE - Electricity Demand Forecasting & Telemetry Dashboard.

Tagline: Predict Demand. Understand Patterns. Stay Ahead.
Professional high-contrast operational analytics and load forecasting interface.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import (
    clean_and_resample,
    data_summary,
    generate_synthetic,
    get_data,
    load_raw,
)
from src.features import build_features
from src.insights import (
    data_driven_observations,
    hourly_profile,
    peak_summary,
    top_hours,
    weekday_profile,
)
from src.models import evaluate, forecast_next_24

# Page configuration (No emojis, simple and clean)
st.set_page_config(
    page_title="WATTWISE | Telemetry Load Forecaster",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Telemetry & Command Center CSS (inspired by motorsport & grid telemetry consoles)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

    /* Global application canvas: Deep matte obsidian */
    .stApp {
        background-color: #08090B !important;
        color: #F4F4F5;
        font-family: 'Inter', sans-serif;
    }
    
    /* Typography Overrides */
    h1, h2, h3, h4 {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }
    
    /* Telemetry Instrument Metric Cards */
    div[data-testid="stMetric"], div.css-1r6slb0, div.css-12w0qpk {
        background: #0E1217 !important;
        border: 1px solid #1E232B !important;
        border-top: 2px solid #14B8A6 !important;
        border-radius: 4px !important;
        padding: 14px 18px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45) !important;
        transition: border-color 0.2s ease, transform 0.2s ease !important;
    }
    
    div[data-testid="stMetric"]:hover {
        border-top-color: #2DD4BF !important;
        border-color: #2D3748 !important;
        transform: translateY(-1px);
    }
    
    div[data-testid="stMetricValue"] > div {
        color: #FFFFFF !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        letter-spacing: -0.01em !important;
        font-feature-settings: "tnum";
        font-variant-numeric: tabular-nums;
    }
    
    div[data-testid="stMetricLabel"] > div {
        color: #71717A !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }
    
    div[data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
    }
    
    /* Telemetry Panel Box */
    .telemetry-card {
        background: #0E1217 !important;
        border: 1px solid #1E232B !important;
        border-radius: 4px !important;
        padding: 16px 20px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
    }
    
    .telemetry-card-accent {
        border-left: 3px solid #14B8A6 !important;
    }
    
    /* Tactical Pill Badges */
    .telemetry-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(20, 184, 166, 0.12);
        color: #14B8A6;
        border: 1px solid rgba(20, 184, 166, 0.35);
        border-radius: 3px;
        padding: 2px 8px;
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    
    .telemetry-pill-active {
        background: rgba(163, 230, 53, 0.12);
        color: #A3E635;
        border: 1px solid rgba(163, 230, 53, 0.35);
    }
    
    /* Pulsing status dot */
    .pulse-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #14B8A6;
        box-shadow: 0 0 6px #14B8A6;
    }
    
    .pulse-dot-active {
        background: #A3E635;
        box-shadow: 0 0 6px #A3E635;
    }
    
    /* Left Sidebar: Matte Cockpit Treatment */
    section[data-testid="stSidebar"] {
        background-color: #0A0C0F !important;
        border-right: 1px solid #181C22 !important;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
    }
    
    /* Segmented Navigation Buttons (Classy full-width tactile tabs) */
    div[data-testid="stRadio"] > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        gap: 5px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    div[data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        padding: 10px 14px !important;
        background: #0E1217 !important;
        border: 1px solid #1C2129 !important;
        border-radius: 4px !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
        margin-bottom: 2px !important;
    }
    
    div[data-testid="stRadio"] label:hover {
        background: #141921 !important;
        border-color: rgba(20, 184, 166, 0.5) !important;
    }
    
    div[data-testid="stRadio"] label:hover div[data-testid="stMarkdownContainer"] p {
        color: #14B8A6 !important;
    }
    
    /* Selected navigation button */
    div[data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(90deg, rgba(20, 184, 166, 0.16) 0%, rgba(20, 184, 166, 0.03) 100%) !important;
        border-color: #14B8A6 !important;
        border-left: 3px solid #14B8A6 !important;
        box-shadow: 0 0 12px rgba(20, 184, 166, 0.15) !important;
    }
    
    div[data-testid="stRadio"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #94A3B8 !important;
        margin: 0 !important;
    }
    
    /* Hide circular native radio input */
    div[data-testid="stRadio"] label div:first-child {
        display: none !important;
    }
    
    /* Table & Dataframe containers */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1E232B !important;
        border-radius: 4px !important;
    }
    
    /* System inputs & sliders */
    .stSelectbox > div > div {
        background-color: #0E1217 !important;
        border-color: #1E232B !important;
        border-radius: 4px !important;
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Cached data loading
@st.cache_data(show_spinner="Loading telemetry stream...")
def load_cached_dataset() -> Tuple[pd.Series, str]:
    """Load and clean time-series data, cached to avoid re-reading raw files."""
    return get_data()


# Cached model evaluation (Seasonal Naive vs. Random Forest)
@st.cache_data(show_spinner="Computing validation metrics...")
def get_cached_evaluation(series: pd.Series) -> Dict[str, Any]:
    """Train models and compute test partition metrics, cached to avoid retraining."""
    return evaluate(series, test_frac=0.2)


# Cached 24-hour recursive forecast
@st.cache_resource(show_spinner="Computing 24-hour predictive trajectory...")
def get_cached_forecast(_model: Any, series: pd.Series) -> pd.DataFrame:
    """Generate 24-step recursive forecast using the trained model."""
    return forecast_next_24(_model, series)


# Cached insights profiles
@st.cache_data
def get_cached_insights(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.DataFrame, list]:
    """Compute hourly and weekday profile aggregations."""
    hp = hourly_profile(series)
    wp = weekday_profile(series)
    top_h = top_hours(series, n=3)
    obs = data_driven_observations(series)
    return hp, wp, top_h, obs


def render_header(source_label: str):
    """Render top operational brand banner and unit clarity notice."""
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.title("WATTWISE")
        st.markdown(
            "<p style='color: #71717A; font-family: Rajdhani, sans-serif; font-size: 1.05rem; font-weight: 600; letter-spacing: 0.12em; margin-top: -8px; text-transform: uppercase;'>"
            "PREDICT DEMAND // UNDERSTAND PATTERNS // STAY AHEAD"
            "</p>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div style='text-align: right; padding-top: 8px;'>
                <span class='telemetry-pill telemetry-pill-active'>
                    <span class='pulse-dot pulse-dot-active'></span> ACTIVE TELEMETRY
                </span><br/>
                <span style='color: #14B8A6; font-family: JetBrains Mono, monospace; font-size: 0.85rem; font-weight: 500;'>
                    {source_label}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("**UNIT SPECIFICATION:** All values represent **hourly average power (kW)**. kW measures instantaneous rate of energy flow; it is not accumulated electrical energy (kWh).")
    st.divider()


def render_overview(series: pd.Series, forecast_df: pd.DataFrame, eval_res: Dict[str, Any], obs: list):
    """Render Overview tab with primary metrics, 7-day + forecast plot, and key insights."""
    st.subheader("SYSTEM OVERVIEW // LOAD TRAJECTORY")

    latest_ts = series.index[-1]
    latest_val = float(series.iloc[-1])
    mean_forecast = float(forecast_df["forecast_kw"].mean())
    max_idx = forecast_df["forecast_kw"].idxmax()
    peak_val = float(forecast_df.loc[max_idx, "forecast_kw"])
    peak_time = forecast_df.loc[max_idx, "timestamp"]

    rf_mae = eval_res["rf_metrics"]["MAE"]
    rf_rmse = eval_res["rf_metrics"]["RMSE"]

    # 4 Executive Telemetry Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="LATEST RECORDED LOAD",
            value=f"{latest_val:.2f} kW",
            delta=f"TIMESTAMP: {latest_ts.strftime('%Y-%m-%d %H:%M')}",
            delta_color="off",
        )
    with m2:
        st.metric(
            label="PREDICTED 24H MEAN",
            value=f"{mean_forecast:.2f} kW",
            delta=f"SPAN: {forecast_df['timestamp'].iloc[0].strftime('%H:%M')} - {forecast_df['timestamp'].iloc[-1].strftime('%H:%M')}",
            delta_color="off",
        )
    with m3:
        st.metric(
            label="PREDICTED PEAK LOAD",
            value=f"{peak_val:.2f} kW",
            delta=f"PEAK AT: {peak_time.strftime('%Y-%m-%d %H:%M')}",
            delta_color="inverse",
        )
    with m4:
        st.metric(
            label="RANDOM FOREST ACCURACY",
            value=f"MAE: {rf_mae:.3f} kW",
            delta=f"RMSE: {rf_rmse:.3f} kW (OPTIMAL)",
            delta_color="normal",
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Plotly Chart: Last 7 days (168h) actuals + 24h forecast
    recent_history = series.iloc[-168:] if len(series) >= 168 else series

    fig = go.Figure()

    # Historical Line (Teal)
    fig.add_trace(
        go.Scatter(
            x=recent_history.index,
            y=recent_history.values,
            mode="lines",
            name="HISTORICAL ACTUALS (PAST 7 DAYS)",
            line=dict(color="#14B8A6", width=2.0),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>Actual:</b> %{y:.2f} kW<extra></extra>",
        )
    )

    # Forecast Line (Lime dashed)
    bridge_x = [recent_history.index[-1]] + list(forecast_df["timestamp"])
    bridge_y = [recent_history.iloc[-1]] + list(forecast_df["forecast_kw"])

    fig.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_y,
            mode="lines+markers",
            name="24H RECURSIVE FORECAST (RF)",
            line=dict(color="#A3E635", width=2.2, dash="dash"),
            marker=dict(size=5, color="#A3E635"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>Forecast:</b> %{y:.2f} kW<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="HISTORICAL PROFILE (7 DAYS) & PREDICTED TRAJECTORY (24 HOURS)",
            font=dict(family="Rajdhani, sans-serif", size=15, color="#E4E4E7"),
        ),
        xaxis=dict(
            title="TIMELINE",
            gridcolor="rgba(255, 255, 255, 0.04)",
            linecolor="#1E232B",
        ),
        yaxis=dict(
            title="HOURLY AVERAGE POWER (kW)",
            gridcolor="rgba(255, 255, 255, 0.04)",
            linecolor="#1E232B",
        ),
        template="plotly_dark",
        paper_bgcolor="#0E1217",
        plot_bgcolor="#090B0E",
        font=dict(family="Rajdhani, Inter, sans-serif"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=55, b=40),
        height=440,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Telemetry Insight Card
    st.markdown(
        f"""
        <div class='telemetry-card telemetry-card-accent'>
            <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;'>
                <span class='telemetry-pill'>DIURNAL LOAD SYNTHESIS</span>
                <span style='color: #71717A; font-family: JetBrains Mono, monospace; font-size: 0.75rem;'>REAL-TIME DATA CONVERGENCE</span>
            </div>
            <p style='font-size: 0.98rem; line-height: 1.5; color: #E4E4E7; margin: 0;'>{obs[0]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_forecast_tab(series: pd.Series, forecast_df: pd.DataFrame):
    """Render Forecast tab with telemetry inspection and hourly matrix."""
    st.subheader("PREDICTIVE HORIZON // 24-HOUR TRAJECTORY")
    st.markdown("<p style='color: #71717A; margin-top: -8px;'>Multi-step recursive inference using historical lag structure and shifted rolling windows.</p>", unsafe_allow_html=True)

    history_span = st.selectbox(
        "HISTORICAL CONTEXT WINDOW:",
        ["Past 48 Hours", "Past 7 Days", "Past 14 Days"],
        index=1,
    )
    span_hours = 48 if history_span == "Past 48 Hours" else (168 if history_span == "Past 7 Days" else 336)
    recent_history = series.iloc[-span_hours:] if len(series) >= span_hours else series

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent_history.index,
            y=recent_history.values,
            mode="lines",
            name="ACTUAL LOAD (kW)",
            line=dict(color="#14B8A6", width=2),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Actual: %{y:.3f} kW<extra></extra>",
        )
    )

    bridge_x = [recent_history.index[-1]] + list(forecast_df["timestamp"])
    bridge_y = [recent_history.iloc[-1]] + list(forecast_df["forecast_kw"])

    fig.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_y,
            mode="lines+markers",
            name="PREDICTED LOAD (kW)",
            line=dict(color="#A3E635", width=2.2, dash="dash"),
            marker=dict(size=5, color="#A3E635"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Forecast: %{y:.3f} kW<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"TELEMETRY COMPARISON: {history_span.upper()} VS. UPCOMING 24 HOURS",
            font=dict(family="Rajdhani, sans-serif", size=15, color="#E4E4E7"),
        ),
        xaxis=dict(title="TIMESTAMP", gridcolor="rgba(255, 255, 255, 0.04)", linecolor="#1E232B"),
        yaxis=dict(title="HOURLY AVERAGE POWER (kW)", gridcolor="rgba(255, 255, 255, 0.04)", linecolor="#1E232B"),
        template="plotly_dark",
        paper_bgcolor="#0E1217",
        plot_bgcolor="#090B0E",
        font=dict(family="Rajdhani, Inter, sans-serif"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=55, b=40),
        height=440,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([0.55, 0.45])

    with col1:
        st.markdown("#### HOURLY FORECAST MATRIX (24-STEP)")
        display_df = forecast_df.copy()
        display_df["Step"] = [f"+{i+1}h" for i in range(len(display_df))]
        display_df["Timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        display_df["Forecast Demand (kW)"] = display_df["forecast_kw"].round(3)
        st.dataframe(
            display_df[["Step", "Timestamp", "Forecast Demand (kW)"]],
            hide_index=True,
            use_container_width=True,
            height=380,
        )

    with col2:
        st.markdown("#### TARGET HOUR TELEMETRY INSPECTOR")
        st.markdown("<p style='color: #71717A; margin-top: -8px;'>Isolate and inspect point predictions along the forecast timeline:</p>", unsafe_allow_html=True)
        ts_options = [ts.strftime("%Y-%m-%d %H:%M") for ts in forecast_df["timestamp"]]
        selected_label = st.select_slider("TARGET HOUR:", options=ts_options, value=ts_options[0])

        sel_idx = ts_options.index(selected_label)
        selected_kw = float(forecast_df["forecast_kw"].iloc[sel_idx])
        selected_dt = forecast_df["timestamp"].iloc[sel_idx]
        delta_to_mean = selected_kw - float(forecast_df["forecast_kw"].mean())

        st.metric(
            label=f"PROJECTED LOAD // {selected_label}",
            value=f"{selected_kw:.3f} kW",
            delta=f"{delta_to_mean:+.3f} kW VS 24H MEAN ({forecast_df['forecast_kw'].mean():.2f} kW)",
            delta_color="normal" if delta_to_mean <= 0 else "inverse",
        )

        st.markdown(
            f"""
            <div class='telemetry-card' style='margin-top: 14px;'>
                <div style='font-family: JetBrains Mono, monospace; font-size: 0.85rem; line-height: 1.8; color: #A1A1AA;'>
                    <div>&bull; DAY OF WEEK: <span style='color: #FFFFFF;'>{selected_dt.strftime('%A').upper()}</span></div>
                    <div>&bull; HOUR OF DAY: <span style='color: #FFFFFF;'>{selected_dt.strftime('%H:00')}</span></div>
                    <div>&bull; HORIZON OFFSET: <span style='color: #14B8A6;'>+{sel_idx + 1} HOURS FORWARD</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_peakwatch_tab(series: pd.Series, forecast_df: pd.DataFrame):
    """Render PeakWatch tab with interactive threshold monitoring and alerts."""
    st.subheader("PEAKWATCH // GRID CAPACITY MONITORING")
    st.markdown("<p style='color: #71717A; margin-top: -8px;'>Automated threshold detection and distribution capacity alerts.</p>", unsafe_allow_html=True)

    hist_p90 = float(np.percentile(series, 90))
    hist_p50 = float(np.percentile(series, 50))
    hist_max = float(series.max())

    threshold = st.slider(
        "OPERATIONAL PEAK THRESHOLD (kW):",
        min_value=round(hist_p50, 2),
        max_value=round(hist_max, 2),
        value=round(hist_p90, 2),
        step=0.05,
        help="Hours with predicted load exceeding this value trigger alert states.",
    )

    summary = peak_summary(forecast_df, threshold)

    c1, c2, c3 = st.columns(3)
    with c1:
        if summary["has_violations"]:
            st.error(f"**CRITICAL THRESHOLD ALERT** | {summary['status']}")
        else:
            st.success(f"**STATUS NOMINAL** | {summary['status']}")
    with c2:
        st.metric(
            label="CONFIGURED THRESHOLD",
            value=f"{threshold:.2f} kW",
            delta=f"90TH PERCENTILE: {hist_p90:.2f} kW",
            delta_color="off",
        )
    with c3:
        st.metric(
            label="FORECAST MAXIMUM LOAD",
            value=f"{summary['peak_kw']:.2f} kW",
            delta=f"AT {summary['peak_time'].strftime('%Y-%m-%d %H:%M')}",
            delta_color="inverse",
        )

    # Plotly Chart with Threshold and Exceedance Markers
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["forecast_kw"],
            mode="lines",
            name="FORECAST DEMAND (kW)",
            line=dict(color="#14B8A6", width=2.5),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Demand: %{y:.3f} kW<extra></extra>",
        )
    )

    fig.add_hline(
        y=threshold,
        line_dash="dot",
        line_color="#F87171",
        annotation_text=f"THRESHOLD ({threshold:.2f} kW)",
        annotation_position="top right",
        annotation_font=dict(family="Rajdhani, sans-serif", color="#F87171"),
    )

    if summary["has_violations"]:
        ex_df = summary["exceeding_df"]
        fig.add_trace(
            go.Scatter(
                x=ex_df["timestamp"],
                y=ex_df["forecast_kw"],
                mode="markers",
                name="THRESHOLD EXCEEDED",
                marker=dict(color="#EF4444", size=10, symbol="circle"),
                hovertemplate="<b>EXCEEDANCE</b><br>%{x|%Y-%m-%d %H:%M}<br>Peak Load: %{y:.3f} kW<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="24-HOUR LOAD TRAJECTORY VS. OPERATIONAL THRESHOLD",
            font=dict(family="Rajdhani, sans-serif", size=15, color="#E4E4E7"),
        ),
        xaxis=dict(title="TIMESTAMP", gridcolor="rgba(255, 255, 255, 0.04)", linecolor="#1E232B"),
        yaxis=dict(title="HOURLY AVERAGE POWER (kW)", gridcolor="rgba(255, 255, 255, 0.04)", linecolor="#1E232B"),
        template="plotly_dark",
        paper_bgcolor="#0E1217",
        plot_bgcolor="#090B0E",
        font=dict(family="Rajdhani, Inter, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    if summary["has_violations"]:
        st.markdown("#### CAPACITY EXCEEDANCE DETAIL")
        viol_display = summary["exceeding_df"].copy()
        viol_display["Timestamp"] = viol_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        viol_display["Predicted Load (kW)"] = viol_display["forecast_kw"].round(3)
        viol_display["Excess Over Threshold (kW)"] = (viol_display["forecast_kw"] - threshold).round(3)
        st.dataframe(
            viol_display[["Timestamp", "Predicted Load (kW)", "Excess Over Threshold (kW)"]],
            hide_index=True,
            use_container_width=True,
        )

    st.caption("**OPERATIONAL PROTOCOL:** Load predictions are statistical estimates derived from historical telemetry; they are operational guidelines rather than guaranteed physical grid events.")


def render_model_arena_tab(eval_res: Dict[str, Any]):
    """Render Model Arena tab comparing Seasonal Naive baseline vs. Random Forest."""
    st.subheader("MODEL ARENA // BENCHMARK VALIDATION")
    st.markdown("<p style='color: #71717A; margin-top: -8px;'>Empirical validation on chronological 20% holdout partition (no shuffling, zero leakage).</p>", unsafe_allow_html=True)

    base_m = eval_res["baseline_metrics"]
    rf_m = eval_res["rf_metrics"]

    # Best Model Banner
    st.success(
        f"**OPTIMAL ARCHITECTURE: {eval_res['best_model'].upper()}** -- "
        f"Random Forest reduced RMSE to **{rf_m['RMSE']:.4f} kW** versus **{base_m['RMSE']:.4f} kW** on baseline."
    )

    # Metrics Matrix Table
    comp_df = pd.DataFrame({
        "Model Architecture": [
            "Baseline: Seasonal Naive (24h Persistence)",
            "Machine Learning: Random Forest (200 Trees, Depth 12)",
        ],
        "MAE (kW)": [base_m["MAE"], rf_m["MAE"]],
        "RMSE (kW)": [base_m["RMSE"], rf_m["RMSE"]],
        "MAPE (%)": [f"{base_m['MAPE']:.2f}%", f"{rf_m['MAPE']:.2f}%"],
    })

    st.markdown("#### EMPIRICAL BENCHMARK MATRIX")
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # Window Selection
    test_idx = eval_res["test_index"]
    window_choice = st.selectbox(
        "TEST EVALUATION WINDOW:",
        ["Last 3 Days of Test Set", "Last 7 Days of Test Set", "Last 14 Days of Test Set"],
        index=1,
    )
    window_hours = 72 if window_choice == "Last 3 Days of Test Set" else (168 if window_choice == "Last 7 Days of Test Set" else 336)

    sub_slice = slice(-window_hours, None)
    w_idx = test_idx[sub_slice]
    w_actual = eval_res["y_true"][sub_slice]
    w_base = eval_res["baseline_preds"][sub_slice]
    w_rf = eval_res["rf_preds"][sub_slice]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=w_idx,
            y=w_actual,
            mode="lines",
            name="GROUND TRUTH ACTUAL (kW)",
            line=dict(color="#94A3B8", width=2),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Actual: %{y:.2f} kW<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=w_idx,
            y=w_base,
            mode="lines",
            name="BASELINE (t-24h)",
            line=dict(color="#F59E0B", width=1.8, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Baseline: %{y:.2f} kW<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=w_idx,
            y=w_rf,
            mode="lines",
            name="RANDOM FOREST",
            line=dict(color="#14B8A6", width=2.2),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>RF: %{y:.2f} kW<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"HOLDOUT TEST TRAJECTORY: ACTUAL VS. BASELINE VS. RANDOM FOREST ({window_choice.upper()})",
            font=dict(family="Rajdhani, sans-serif", size=15, color="#E4E4E7"),
        ),
        xaxis=dict(title="TIMESTAMP", gridcolor="rgba(255, 255, 255, 0.04)", linecolor="#1E232B"),
        yaxis=dict(title="HOURLY AVERAGE POWER (kW)", gridcolor="rgba(255, 255, 255, 0.04)", linecolor="#1E232B"),
        template="plotly_dark",
        paper_bgcolor="#0E1217",
        plot_bgcolor="#090B0E",
        font=dict(family="Rajdhani, Inter, sans-serif"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=55, b=40),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bar chart comparing errors
    err_fig = go.Figure()
    err_fig.add_trace(
        go.Bar(
            name="MAE (kW)",
            x=["Seasonal Naive", "Random Forest"],
            y=[base_m["MAE"], rf_m["MAE"]],
            marker_color="#14B8A6",
            text=[f"{base_m['MAE']:.3f}", f"{rf_m['MAE']:.3f}"],
            textposition="auto",
        )
    )
    err_fig.add_trace(
        go.Bar(
            name="RMSE (kW)",
            x=["Seasonal Naive", "Random Forest"],
            y=[base_m["RMSE"], rf_m["RMSE"]],
            marker_color="#A3E635",
            text=[f"{base_m['RMSE']:.3f}", f"{rf_m['RMSE']:.3f}"],
            textposition="auto",
        )
    )
    err_fig.update_layout(
        barmode="group",
        title=dict(
            text="MODEL ERROR BENCHMARK (LOWER REPRESENTS SUPERIOR GENERALIZATION)",
            font=dict(family="Rajdhani, sans-serif", size=14, color="#E4E4E7"),
        ),
        yaxis=dict(title="ERROR (kW)", gridcolor="rgba(255, 255, 255, 0.04)", linecolor="#1E232B"),
        xaxis=dict(linecolor="#1E232B"),
        template="plotly_dark",
        paper_bgcolor="#0E1217",
        plot_bgcolor="#090B0E",
        font=dict(family="Rajdhani, Inter, sans-serif"),
        height=320,
        margin=dict(l=40, r=40, t=50, b=30),
    )
    st.plotly_chart(err_fig, use_container_width=True)


def render_pattern_lens_tab(hp: pd.Series, wp: pd.Series, top_h: pd.DataFrame, obs: list):
    """Render Pattern Lens tab with diurnal profiles and telemetry analytics."""
    st.subheader("PATTERN LENS // BEHAVIORAL LOAD DYNAMICS")
    st.markdown("<p style='color: #71717A; margin-top: -8px;'>Diurnal cycles, weekly rhythms, and temporal demand characteristics.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        h_fig = go.Figure()
        h_fig.add_trace(
            go.Bar(
                x=[f"{h:02d}:00" for h in hp.index],
                y=hp.values,
                marker_color="#14B8A6",
                hovertemplate="Hour %{x}<br>Average Demand: %{y:.3f} kW<extra></extra>",
            )
        )
        h_fig.update_layout(
            title=dict(
                text="DIURNAL LOAD PROFILE (HOURLY AVERAGE kW)",
                font=dict(family="Rajdhani, sans-serif", size=14, color="#E4E4E7"),
            ),
            xaxis=dict(title="HOUR OF DAY", gridcolor="rgba(255, 255, 255, 0.04)", linecolor="#1E232B"),
            yaxis=dict(title="HOURLY AVERAGE POWER (kW)", gridcolor="rgba(255, 255, 255, 0.04)", linecolor="#1E232B"),
            template="plotly_dark",
            paper_bgcolor="#0E1217",
            plot_bgcolor="#090B0E",
            font=dict(family="Rajdhani, Inter, sans-serif"),
            margin=dict(l=40, r=40, t=50, b=40),
            height=340,
        )
        st.plotly_chart(h_fig, use_container_width=True)

    with col2:
        w_fig = go.Figure()
        w_fig.add_trace(
            go.Bar(
                x=wp.index,
                y=wp.values,
                marker_color="#A3E635",
                hovertemplate="%{x}<br>Average Demand: %{y:.3f} kW<extra></extra>",
            )
        )
        w_fig.update_layout(
            title=dict(
                text="WEEKLY RHYTHM (DAY OF WEEK AVERAGE kW)",
                font=dict(family="Rajdhani, sans-serif", size=14, color="#E4E4E7"),
            ),
            xaxis=dict(title="DAY OF WEEK", gridcolor="rgba(255, 255, 255, 0.04)", linecolor="#1E232B"),
            yaxis=dict(title="HOURLY AVERAGE POWER (kW)", gridcolor="rgba(255, 255, 255, 0.04)", linecolor="#1E232B"),
            template="plotly_dark",
            paper_bgcolor="#0E1217",
            plot_bgcolor="#090B0E",
            font=dict(family="Rajdhani, Inter, sans-serif"),
            margin=dict(l=40, r=40, t=50, b=40),
            height=340,
        )
        st.plotly_chart(w_fig, use_container_width=True)

    col3, col4 = st.columns([0.4, 0.6])

    with col3:
        st.markdown("#### HISTORICAL PEAK DISTRIBUTION")
        top_display = top_h.copy()
        top_display["Hour"] = [f"{int(h):02d}:00" for h in top_display["hour"]]
        top_display["Avg Demand (kW)"] = top_display["avg_kw"].round(3)
        st.dataframe(
            top_display[["rank", "Hour", "Avg Demand (kW)"]].rename(columns={"rank": "Rank"}),
            hide_index=True,
            use_container_width=True,
        )

    with col4:
        st.markdown("#### ANALYTIC LOAD SYNTHESIS")
        for i, observation in enumerate(obs, 1):
            st.markdown(
                f"""
                <div class='telemetry-card telemetry-card-accent'>
                    <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 4px;'>
                        <span class='telemetry-pill'>OBSERVATION #{i}</span>
                    </div>
                    <p style='font-size: 0.95rem; line-height: 1.45; color: #E4E4E7; margin: 0;'>{observation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main():
    # Sidebar setup
    st.sidebar.markdown(
        "<h2 style='color: #FFFFFF; font-family: Rajdhani, sans-serif; letter-spacing: 0.08em; margin-bottom: 0;'>WATTWISE</h2>"
        "<p style='color: #71717A; font-family: Rajdhani, sans-serif; font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase;'>TELEMETRY FORECASTER</p>",
        unsafe_allow_html=True,
    )

    data_source_mode = st.sidebar.selectbox(
        "DATA SOURCE MODE:",
        ["Auto (UCI Dataset / Fallback)", "Upload Custom CSV / TXT", "Synthetic Demo"],
        index=0,
        help="Select telemetry source: primary dataset, custom upload, or benchmark.",
    )

    series = None
    source_label = ""

    try:
        if data_source_mode == "Auto (UCI Dataset / Fallback)":
            series, source_label = load_cached_dataset()
        elif data_source_mode == "Synthetic Demo":
            series = generate_synthetic(days=365)
            source_label = "SYNTHETIC DEMO"
        elif data_source_mode == "Upload Custom CSV / TXT":
            uploaded_file = st.sidebar.file_uploader(
                "Upload Electricity Dataset (CSV / TXT):",
                type=["csv", "txt"],
                help="File must contain Date + Time or datetime, plus Global_active_power.",
            )
            if uploaded_file is None:
                st.info("**FILE REQUIRED:** Please upload a dataset file in the sidebar to proceed. Expected columns: `Date` and `Time` (or `datetime`) and `Global_active_power`.")
                return

            if uploaded_file.size == 0:
                st.error("**DATASET INGESTION ERROR:** The uploaded file is completely empty (0 bytes).")
                return

            try:
                temp_upload_path = Path("data") / f"temp_upload_{uploaded_file.name}"
                temp_upload_path.write_bytes(uploaded_file.getbuffer())
                raw_df = load_raw(temp_upload_path)
                series = clean_and_resample(raw_df)
                if temp_upload_path.exists():
                    temp_upload_path.unlink()

                if series.empty:
                    st.error("**DATASET INGESTION ERROR:** No valid non-negative readings found after cleaning and resampling.")
                    return
                source_label = f"Uploaded File: {uploaded_file.name}"
            except Exception as read_err:
                st.error(f"**DATASET INGESTION ERROR:** {read_err}")
                return

    except Exception as exc:
        st.error(f"**DATASET INGESTION ERROR:** Failed to load dataset: {exc}")
        return

    st.sidebar.markdown(
        f"<div style='margin-top: -6px; margin-bottom: 12px; font-family: JetBrains Mono, monospace; font-size: 0.78rem; color: #71717A;'>"
        f"SOURCE: <span style='color: #14B8A6;'>{source_label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if source_label == "SYNTHETIC DEMO":
        st.sidebar.warning("**SYSTEM NOTICE:** Running in SYNTHETIC DEMO mode.")
        st.warning("**SYSTEM NOTICE:** WATTWISE is running in **SYNTHETIC DEMO** mode. Real UCI dataset was not detected in 'data/'. Displaying synthetic benchmark telemetry.")

    nav_page = st.sidebar.radio(
        "TELEMETRY CONSOLE:",
        ["Overview", "Forecast", "PeakWatch", "Model Arena", "Pattern Lens"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        "<div style='font-family: Rajdhani, sans-serif; font-size: 0.85rem; color: #71717A; line-height: 1.6;'>"
        "<b style='color: #E4E4E7;'>COLLEGE MINI-PROJECT</b><br/>"
        "Model: Random Forest (200 Trees)<br/>"
        "Partition: Chronological Split<br/>"
        "Integrity: Zero Data Leakage"
        "</div>",
        unsafe_allow_html=True,
    )

    render_header(source_label)

    # Compute and cache evaluation, forecasts, and pattern profiles
    try:
        eval_res = get_cached_evaluation(series)
        forecast_df = get_cached_forecast(eval_res["model"], series)
        hp, wp, top_h, obs = get_cached_insights(series)
    except Exception as exc:
        st.error(f"**MODEL INFERENCE ERROR:** {exc}")
        return

    # Route navigation
    if nav_page == "Overview":
        render_overview(series, forecast_df, eval_res, obs)
    elif nav_page == "Forecast":
        render_forecast_tab(series, forecast_df)
    elif nav_page == "PeakWatch":
        render_peakwatch_tab(series, forecast_df)
    elif nav_page == "Model Arena":
        render_model_arena_tab(eval_res)
    elif nav_page == "Pattern Lens":
        render_pattern_lens_tab(hp, wp, top_h, obs)


if __name__ == "__main__":
    main()
