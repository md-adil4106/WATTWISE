"""WATTWISE - Streamlit Electricity-Demand Forecasting Dashboard.

A streamlined, reliable electricity demand forecasting application for predicting demand,
analyzing consumption patterns, and planning grid or facility loads.
"""

import streamlit as st

st.set_page_config(
    page_title="WATTWISE",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    st.title("⚡ WATTWISE")
    st.subheader("WATTWISE: Predict Demand. Understand Patterns. Stay Ahead.")
    
    st.markdown(
        """
        Welcome to **WATTWISE**, an electricity-demand forecasting dashboard designed to deliver
        clear, reliable short-term and medium-term load predictions.
        
        ### Project Status
        - **Phase 0:** Project Scaffolding & Setup (Completed)
        - **Phase 1:** Data Ingestion & Hourly Resampling (Planned)
        - **Phase 2:** Feature Engineering & Leakage Prevention (Planned)
        - **Phase 3:** Baseline & ML Forecasting Models (Planned)
        - **Phase 4:** Insights, Peak Detection & Interactive Dashboard (Planned)
        """
    )
    
    with st.sidebar:
        st.header("⚡ WATTWISE Control")
        st.info("System initialized. Core modules scaffolded.")


if __name__ == "__main__":
    main()
