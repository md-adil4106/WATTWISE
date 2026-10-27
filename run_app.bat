@echo off
title WATTWISE Telemetry Dashboard
cd /d "%~dp0"
echo ===================================================
echo   WATTWISE: Electricity Consumption Forecasting
echo ===================================================
echo Launching Streamlit dashboard on http://localhost:8501 ...
echo Press Ctrl+C in this terminal to stop the server.
echo.
call .venv\Scripts\streamlit.exe run app.py
pause
