"""WATTWISE - Electricity Demand Forecasting Dashboard.

Tagline: Predict Demand. Understand Patterns. Stay Ahead.
Interactive Streamlit application for time-series analytics, load forecasting,
peak demand monitoring, and model evaluation.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import data_summary, get_data
from src.features import build_features
from src.insights import (
    data_driven_observations,
    hourly_profile,
    peak_summary,
    top_hours,
    weekday_profile,
)
from src.models import evaluate, forecast_next_24

# Page configuration
st.set_page_config(
    page_title="WATTWISE | Electricity Demand Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Dark Charcoal (#0F1412), Teal (#14B8A6), and Lime (#A3E635) styling
st.markdown(
    """
    <style>
    /* Global font & background enhancements */
    .stApp {
        background-color: #0F1412;
        color: #F3F4F6;
    }
    
    /* Card containers */
    div[data-testid="stMetric"], div.css-1r6slb0, div.css-12w0qpk {
        background: #18221E;
        border: 1px solid rgba(20, 184, 166, 0.25);
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Metric typography */
    div[data-testid="stMetricValue"] > div {
        color: #14B8A6 !important;
        font-weight: 700 !important;
        font-size: 1.85rem !important;
    }
    
    div[data-testid="stMetricLabel"] > div {
        color: #9CA3AF !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Custom callout boxes */
    .wattwise-card {
        background-color: #18221E;
        border: 1px solid rgba(20, 184, 166, 0.3);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    
    .wattwise-badge {
        display: inline-block;
        background: rgba(163, 230, 53, 0.15);
        color: #A3E635;
        border: 1px solid rgba(163, 230, 53, 0.35);
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #131A17;
        border-right: 1px solid rgba(20, 184, 166, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Cached data loading
@st.cache_data(show_spinner="Loading electricity consumption time series...")
def load_cached_dataset() -> Tuple[pd.Series, str]:
    """Load and clean time-series data, cached to avoid re-reading raw files."""
    return get_data()


# Cached model evaluation (Seasonal Naive vs. Random Forest)
@st.cache_data(show_spinner="Training forecaster & evaluating test metrics...")
def get_cached_evaluation(series: pd.Series) -> Dict[str, Any]:
    """Train models and compute test partition metrics, cached to avoid retraining on UI interactions."""
    return evaluate(series, test_frac=0.2)


# Cached 24-hour recursive forecast
@st.cache_resource(show_spinner="Generating 24-hour recursive forecast...")
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
    """Render top application brand banner and unit clarity disclaimer."""
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.title("⚡ WATTWISE")
        st.markdown(
            "#### *Predict Demand. Understand Patterns. Stay Ahead.*\n"
            "<small style='color: #9CA3AF;'>Electricity-Demand Forecasting & Operational Analytics Dashboard</small>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div style='text-align: right; padding-top: 10px;'>
                <span class='wattwise-badge'>Active Source</span><br/>
                <span style='color: #14B8A6; font-weight: 600;'>{source_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Unit note banner
    st.caption("ℹ️ **Unit Notice:** All values represent **hourly average power (kW)**. kW measures instantaneous rate of energy flow; it is not accumulated electrical energy (kWh).")
    st.divider()


def render_overview(series: pd.Series, forecast_df: pd.DataFrame, eval_res: Dict[str, Any], obs: list):
    """Render Overview tab with primary metrics, 7-day + forecast plot, and key insights."""
    st.subheader("System Overview & Load Trajectory")

    latest_ts = series.index[-1]
    latest_val = float(series.iloc[-1])
    mean_forecast = float(forecast_df["forecast_kw"].mean())
    max_idx = forecast_df["forecast_kw"].idxmax()
    peak_val = float(forecast_df.loc[max_idx, "forecast_kw"])
    peak_time = forecast_df.loc[max_idx, "timestamp"]

    rf_mae = eval_res["rf_metrics"]["MAE"]
    rf_rmse = eval_res["rf_metrics"]["RMSE"]

    # 4 Executive KPI Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Latest Recorded Power",
            value=f"{latest_val:.2f} kW",
            delta=f"Timestamp: {latest_ts.strftime('%b %d, %H:%M')}",
            delta_color="off",
        )
    with m2:
        st.metric(
            label="Predicted 24h Mean",
            value=f"{mean_forecast:.2f} kW",
            delta=f"Span: {forecast_df['timestamp'].iloc[0].strftime('%H:%M')} - {forecast_df['timestamp'].iloc[-1].strftime('%H:%M')}",
            delta_color="off",
        )
    with m3:
        st.metric(
            label="Predicted Peak Load",
            value=f"{peak_val:.2f} kW",
            delta=f"Peak at {peak_time.strftime('%b %d, %H:%M')}",
            delta_color="inverse",
        )
    with m4:
        st.metric(
            label="Random Forest Accuracy",
            value=f"MAE: {rf_mae:.3f} kW",
            delta=f"RMSE: {rf_rmse:.3f} kW (Best Model)",
            delta_color="normal",
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Prominent Plotly chart: Last 7 days (168h) of actuals + 24h forecast
    recent_history = series.iloc[-168:] if len(series) >= 168 else series

    fig = go.Figure()

    # Historical line (Teal)
    fig.add_trace(
        go.Scatter(
            x=recent_history.index,
            y=recent_history.values,
            mode="lines",
            name="Recorded Demand (Past 7 Days)",
            line=dict(color="#14B8A6", width=2.2),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>Actual:</b> %{y:.2f} kW<extra></extra>",
        )
    )

    # Forecast line (Lime dashed)
    # Connect last historical point to first forecast point for seamless continuity
    bridge_x = [recent_history.index[-1]] + list(forecast_df["timestamp"])
    bridge_y = [recent_history.iloc[-1]] + list(forecast_df["forecast_kw"])

    fig.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_y,
            mode="lines+markers",
            name="24-Hour Forecast (Recursive RF)",
            line=dict(color="#A3E635", width=2.5, dash="dash"),
            marker=dict(size=6, color="#A3E635"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>Forecast:</b> %{y:.2f} kW<extra></extra>",
        )
    )

    fig.update_layout(
        title="7-Day Historical Demand Profile & Upcoming 24-Hour Forecast",
        xaxis_title="Timeline",
        yaxis_title="Hourly Average Power (kW)",
        template="plotly_dark",
        paper_bgcolor="#18221E",
        plot_bgcolor="#18221E",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Data-driven insight card
    st.markdown(
        f"""
        <div class='wattwise-card'>
            <span class='wattwise-badge'>💡 Operational Demand Insight</span><br/>
            <span style='font-size: 1.05rem; line-height: 1.5;'>{obs[0]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_forecast_tab(series: pd.Series, forecast_df: pd.DataFrame):
    """Render Forecast tab with interactive timeline, inspection slider, and tabular breakdown."""
    st.subheader("24-Hour Predictive Horizon")
    st.markdown("Recursive multi-step electricity load forecasts generated using lag structures and shifted rolling windows.")

    history_span = st.selectbox(
        "Historical Context Display Window:",
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
            name="Historical Actuals (kW)",
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
            name="Forecast Next 24h (kW)",
            line=dict(color="#A3E635", width=2.5, dash="dash"),
            marker=dict(size=5, color="#A3E635"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Forecast: %{y:.3f} kW<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Actual History ({history_span}) vs. 24-Hour Ahead Predictions",
        xaxis_title="Timestamp",
        yaxis_title="Hourly Average Power (kW)",
        template="plotly_dark",
        paper_bgcolor="#18221E",
        plot_bgcolor="#18221E",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=440,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([0.55, 0.45])

    with col1:
        st.markdown("#### 🕒 Step-by-Step Hourly Forecast Table")
        display_df = forecast_df.copy()
        display_df["Hour Ahead"] = [f"+{i+1}h" for i in range(len(display_df))]
        display_df["Timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        display_df["Forecast Demand (kW)"] = display_df["forecast_kw"].round(3)
        st.dataframe(
            display_df[["Hour Ahead", "Timestamp", "Forecast Demand (kW)"]],
            hide_index=True,
            use_container_width=True,
            height=380,
        )

    with col2:
        st.markdown("#### 🔍 Timestamp Inspector")
        st.markdown("Select any specific hour in the forecast horizon to inspect its predicted load:")
        ts_options = [ts.strftime("%Y-%m-%d %H:%M") for ts in forecast_df["timestamp"]]
        selected_label = st.select_slider("Choose Target Hour:", options=ts_options, value=ts_options[0])

        sel_idx = ts_options.index(selected_label)
        selected_kw = float(forecast_df["forecast_kw"].iloc[sel_idx])
        selected_dt = forecast_df["timestamp"].iloc[sel_idx]
        delta_to_mean = selected_kw - float(forecast_df["forecast_kw"].mean())

        st.metric(
            label=f"Predicted Load for {selected_label}",
            value=f"{selected_kw:.3f} kW",
            delta=f"{delta_to_mean:+.3f} kW vs 24h Mean ({forecast_df['forecast_kw'].mean():.2f} kW)",
            delta_color="normal" if delta_to_mean <= 0 else "inverse",
        )

        st.info(
            f"**Inspection Details:**\n"
            f"- **Day of Week:** {selected_dt.strftime('%A')}\n"
            f"- **Hour of Day:** {selected_dt.strftime('%H:00')}\n"
            f"- **Horizon Step:** +{sel_idx + 1} hours from current observation"
        )


def render_peakwatch_tab(series: pd.Series, forecast_df: pd.DataFrame):
    """Render PeakWatch tab with interactive threshold slider, violation highlights, and alerts."""
    st.subheader("PeakWatch: Demand Threshold Monitoring & Alerts")
    st.markdown("Identify impending capacity stress and configure operational load thresholds.")

    # Threshold defaults to 90th percentile of historical demand
    hist_p90 = float(np.percentile(series, 90))
    hist_p50 = float(np.percentile(series, 50))
    hist_max = float(series.max())

    threshold = st.slider(
        "Set Operational Peak Threshold (kW):",
        min_value=round(hist_p50, 2),
        max_value=round(hist_max, 2),
        value=round(hist_p90, 2),
        step=0.05,
        help="Hours with predicted demand exceeding this value are flagged as potential peak events.",
    )

    summary = peak_summary(forecast_df, threshold)

    c1, c2, c3 = st.columns(3)
    with c1:
        if summary["has_violations"]:
            st.error(f"🚨 **{summary['status']}**")
        else:
            st.success(f"✅ **{summary['status']}**")
    with c2:
        st.metric(
            label="Configured Threshold",
            value=f"{threshold:.2f} kW",
            delta=f"90th Percentile: {hist_p90:.2f} kW",
            delta_color="off",
        )
    with c3:
        st.metric(
            label="Forecast Maximum Load",
            value=f"{summary['peak_kw']:.2f} kW",
            delta=f"At {summary['peak_time'].strftime('%b %d, %H:%M')}",
            delta_color="inverse",
        )

    # Chart highlighting threshold line and exceeding points
    fig = go.Figure()

    # Forecast curve
    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["forecast_kw"],
            mode="lines",
            name="24h Forecast Load (kW)",
            line=dict(color="#14B8A6", width=2.5),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Demand: %{y:.3f} kW<extra></extra>",
        )
    )

    # Threshold horizontal reference line
    fig.add_hline(
        y=threshold,
        line_dash="dot",
        line_color="#F87171",
        annotation_text=f"Threshold ({threshold:.2f} kW)",
        annotation_position="top right",
    )

    # Exceeding violation markers (if any)
    if summary["has_violations"]:
        ex_df = summary["exceeding_df"]
        fig.add_trace(
            go.Scatter(
                x=ex_df["timestamp"],
                y=ex_df["forecast_kw"],
                mode="markers",
                name="Threshold Exceeded",
                marker=dict(color="#EF4444", size=11, symbol="circle"),
                hovertemplate="<b>VIOLATION</b><br>%{x|%Y-%m-%d %H:%M}<br>Exceeding Load: %{y:.3f} kW<extra></extra>",
            )
        )

    fig.update_layout(
        title="24-Hour Forecast vs. Operational Peak Threshold",
        xaxis_title="Forecast Timestamp",
        yaxis_title="Hourly Average Power (kW)",
        template="plotly_dark",
        paper_bgcolor="#18221E",
        plot_bgcolor="#18221E",
        margin=dict(l=40, r=40, t=50, b=40),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    if summary["has_violations"]:
        st.markdown("#### ⚠️ Exceeding Hours Detail")
        viol_display = summary["exceeding_df"].copy()
        viol_display["Timestamp"] = viol_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        viol_display["Predicted Load (kW)"] = viol_display["forecast_kw"].round(3)
        viol_display["Excess Over Threshold (kW)"] = (viol_display["forecast_kw"] - threshold).round(3)
        st.dataframe(
            viol_display[["Timestamp", "Predicted Load (kW)", "Excess Over Threshold (kW)"]],
            hide_index=True,
            use_container_width=True,
        )

    # Required operational disclaimer
    st.caption("⚠️ **Operational Disclaimer:** Load predictions are statistical estimates derived from historical patterns; they are operational guidelines rather than guaranteed physical grid events.")


def render_model_arena_tab(eval_res: Dict[str, Any]):
    """Render Model Arena tab comparing Seasonal Naive baseline vs. Random Forest on identical test timestamps."""
    st.subheader("Model Arena: Rigorous Chronological Benchmark")
    st.markdown("Both models are evaluated on the exact same chronological 20% test partition without shuffling or leakage.")

    base_m = eval_res["baseline_metrics"]
    rf_m = eval_res["rf_metrics"]

    # Best model banner
    st.success(
        f"🏆 **Champion Model by RMSE: {eval_res['best_model']}** — "
        f"Random Forest reduced RMSE to **{rf_m['RMSE']:.4f} kW** versus **{base_m['RMSE']:.4f} kW** on the baseline."
    )

    # Metrics comparison table
    comp_df = pd.DataFrame({
        "Model Architecture": [
            "Baseline: Seasonal Naive (24h Persistence)",
            "Machine Learning: Random Forest (200 Trees, Depth 12)",
        ],
        "MAE (kW)": [base_m["MAE"], rf_m["MAE"]],
        "RMSE (kW)": [base_m["RMSE"], rf_m["RMSE"]],
        "MAPE (%)": [f"{base_m['MAPE']:.2f}%", f"{rf_m['MAPE']:.2f}%"],
    })

    st.markdown("#### 📊 Evaluation Metrics Table")
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # Test set window selection
    test_idx = eval_res["test_index"]
    window_choice = st.selectbox(
        "Test Horizon Visual Window:",
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
            name="Ground Truth (Actual kW)",
            line=dict(color="#94A3B8", width=2),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Actual: %{y:.2f} kW<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=w_idx,
            y=w_base,
            mode="lines",
            name="Seasonal Naive Baseline (t-24h)",
            line=dict(color="#F59E0B", width=1.8, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Baseline: %{y:.2f} kW<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=w_idx,
            y=w_rf,
            mode="lines",
            name="Random Forest Forecaster",
            line=dict(color="#14B8A6", width=2.2),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>RF: %{y:.2f} kW<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Test Set Comparison: Actual vs. Baseline vs. Random Forest ({window_choice})",
        xaxis_title="Test Partition Timestamp",
        yaxis_title="Hourly Average Power (kW)",
        template="plotly_dark",
        paper_bgcolor="#18221E",
        plot_bgcolor="#18221E",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
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
        title="Model Error Comparison (Lower is Superior)",
        yaxis_title="Error (kW)",
        template="plotly_dark",
        paper_bgcolor="#18221E",
        plot_bgcolor="#18221E",
        height=320,
        margin=dict(l=40, r=40, t=50, b=30),
    )
    st.plotly_chart(err_fig, use_container_width=True)


def render_pattern_lens_tab(hp: pd.Series, wp: pd.Series, top_h: pd.DataFrame, obs: list):
    """Render Pattern Lens tab with diurnal profiles, weekly cycles, top hours, and observations."""
    st.subheader("Pattern Lens: Time-Series Dynamics & Behavioral Signatures")
    st.markdown("Understand demand rhythms across daily hours and weekly cycles.")

    # Hourly Profile Chart
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
            title="Diurnal Demand Profile: Average kW by Hour of Day",
            xaxis_title="Hour of Day",
            yaxis_title="Hourly Average Power (kW)",
            template="plotly_dark",
            paper_bgcolor="#18221E",
            plot_bgcolor="#18221E",
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
            title="Weekly Rhythm: Average kW by Day of Week",
            xaxis_title="Day of Week",
            yaxis_title="Hourly Average Power (kW)",
            template="plotly_dark",
            paper_bgcolor="#18221E",
            plot_bgcolor="#18221E",
            margin=dict(l=40, r=40, t=50, b=40),
            height=340,
        )
        st.plotly_chart(w_fig, use_container_width=True)

    # Top Hours & Data-driven observations
    col3, col4 = st.columns([0.4, 0.6])

    with col3:
        st.markdown("#### 🔝 Peak Demand Hours")
        top_display = top_h.copy()
        top_display["Hour"] = [f"{int(h):02d}:00" for h in top_display["hour"]]
        top_display["Avg Demand (kW)"] = top_display["avg_kw"].round(3)
        st.dataframe(
            top_display[["rank", "Hour", "Avg Demand (kW)"]].rename(columns={"rank": "Rank"}),
            hide_index=True,
            use_container_width=True,
        )

    with col4:
        st.markdown("#### 📋 Data-Driven Takeaways")
        for i, observation in enumerate(obs, 1):
            st.markdown(
                f"""
                <div class='wattwise-card'>
                    <span class='wattwise-badge'>Observation #{i}</span><br/>
                    <span style='font-size: 0.95rem; line-height: 1.4;'>{observation}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main():
    try:
        series, source_label = load_cached_dataset()
    except Exception as exc:
        st.error(
            f"❌ **Failed to load WATTWISE dataset:** {exc}\n\n"
            "Please check that 'data/household_power_consumption.txt' exists or run 'python scripts/download_uci.py'."
        )
        return

    # Sidebar setup
    st.sidebar.title("⚡ WATTWISE")
    st.sidebar.markdown(f"**Data Source:** `{source_label}`")

    # Synthetic Demo Mode Warning on every page if running in fallback mode
    if source_label == "SYNTHETIC DEMO":
        st.sidebar.warning("⚠️ Running in SYNTHETIC DEMO mode.")
        st.warning("⚠️ **Notice:** WATTWISE is running in **SYNTHETIC DEMO** mode. Real UCI dataset was not found; displaying synthetic benchmark data.")

    nav_page = st.sidebar.radio(
        "Navigation Menu:",
        ["Overview", "Forecast", "PeakWatch", "Model Arena", "Pattern Lens"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.info(
        "**College Mini-Project**\n"
        "Electricity Demand Forecasting\n"
        "Model: Random Forest Regressor\n"
        "Evaluation: Strict Chronological Split\n"
        "Zero Data Leakage Enforced"
    )

    render_header(source_label)

    # Compute and cache evaluation, forecasts, and pattern profiles
    try:
        eval_res = get_cached_evaluation(series)
        forecast_df = get_cached_forecast(eval_res["model"], series)
        hp, wp, top_h, obs = get_cached_insights(series)
    except Exception as exc:
        st.error(f"❌ **An error occurred during model evaluation or inference:** {exc}")
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
