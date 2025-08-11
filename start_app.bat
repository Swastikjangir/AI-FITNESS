@echo off
REM AI Fitness Coach - Windows Startup Script
REM This script starts the AI Fitness Coach application

echo.
echo ========================================
echo    AI FITNESS COACH STARTUP SCRIPT
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
python -c "import streamlit, opencv-python, mediapipe, numpy, pandas" >nul 2>&1
if errorlevel 1 (
    echo Installing required dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Start the application
echo.
echo Starting AI Fitness Coach...
echo The application will open in your default web browser
echo If it doesn't open automatically, navigate to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

REM Run the application
python run_app.py --streamlit

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Application stopped with an error
    pause
)

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat 