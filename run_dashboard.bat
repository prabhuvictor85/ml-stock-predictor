@echo off
cd /d C:\Victor\Project\ml-stock-predictor
echo Starting NSE ML Stock Predictor Dashboard...
echo.
echo Dashboard will open in your browser at http://localhost:8501
echo Press Ctrl+C in this window to stop the dashboard.
echo.
.venv\Scripts\streamlit run dashboard.py
pause
