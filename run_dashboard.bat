@echo off
echo Starting NorthBay Living Planning Dashboard...
call venv\Scripts\activate.bat
streamlit run app\app.py
pause
