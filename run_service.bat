@echo off
echo Starting NorthBay Living Scoring API Service...
call venv\Scripts\activate.bat
uvicorn service.main:app --host 127.0.0.1 --port 8000 --reload
pause
