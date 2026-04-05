@echo off
title MT5 Data Pipeline — Automatic Mode
color 0B
cd /d "%~dp0"

echo.
echo  ======================================
echo   MT5 DATA PIPELINE — ONE CLICK START
echo  ======================================
echo.

:: Create venv if missing
if not exist ".venv\Scripts\python.exe" (
    echo  [SETUP] Creating virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo  [ERROR] Python 3.11+ is required. Install from python.org
        pause
        exit /b 1
    )
)

:: Fast dependency check: only install when environment is missing required packages
echo  [SETUP] Checking dependencies...
.venv\Scripts\python.exe -c "import MetaTrader5, polars, rich, yaml" >nul 2>nul
if errorlevel 1 (
    echo  [SETUP] Installing dependencies...
    .venv\Scripts\python.exe -m pip install -q --upgrade pip 2>nul
    .venv\Scripts\python.exe -m pip install -q -e . 2>nul
    if errorlevel 1 (
        echo  [ERROR] Dependency install failed. Check your internet connection.
        pause
        exit /b 1
    )
) else (
    echo  [SETUP] Dependencies already satisfied.
)

:: Launch zero-config orchestrator
echo  [READY] Launching pipeline...
echo.
.venv\Scripts\python.exe -m mt5pipe.tools.super_pipeline_tui
set EXIT_CODE=%ERRORLEVEL%

echo.
if "%EXIT_CODE%"=="0" (
    echo  Pipeline completed successfully.
) else (
    echo  Pipeline exited with code %EXIT_CODE%.
)

pause
exit /b %EXIT_CODE%
