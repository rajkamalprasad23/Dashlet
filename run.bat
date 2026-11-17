@echo off
title Dashlet - Short Video Streaming Prototype
echo ========================================================
echo   Dashlet: Robust Short Video Streaming Prototype
echo   Based on Li et al., SIGCOMM 2022
echo ========================================================
echo.
cd /d "%~dp0server"

echo [1/2] Checking Python requirements...
python -m pip install -r requirements.txt --quiet

echo.
echo [2/2] Launching Dashlet Server on http://localhost:5000 ...
echo Press Ctrl+C in this window to stop the server.
echo.
start http://localhost:5000
python app.py

pause
