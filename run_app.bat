@echo off
title RealFieldAI - Real Measurements, Intelligent Predictions
echo =====================================================================
echo  Menjalankan RealFieldAI
echo  Real Measurement-Based Field Artificial Intelligence
echo  Tagline: Real Measurements, Intelligent Predictions
echo =====================================================================
set "PY_CMD=python"
where python >nul 2>&1
if errorlevel 1 (
    if exist "C:\ProgramData\anaconda3\python.exe" (
        set "PY_CMD=C:\ProgramData\anaconda3\python.exe"
    )
) else (
    python --version >nul 2>&1
    if errorlevel 1 (
        if exist "C:\ProgramData\anaconda3\python.exe" (
            set "PY_CMD=C:\ProgramData\anaconda3\python.exe"
        )
    )
)

"%PY_CMD%" -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Terjadi kendala saat menjalankan aplikasi. Tekan sembarang tombol untuk keluar...
    pause >nul
)
