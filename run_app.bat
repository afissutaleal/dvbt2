@echo off
title DVB-T2 Broadcast AI Propagation & GIS Suite
echo =====================================================================
echo  Menjalankan DVB-T2 Broadcast AI Propagation & GIS Suite...
echo =====================================================================
cd /d "%~dp0"
python -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Terjadi kendala saat menjalankan aplikasi. Tekan sembarang tombol untuk keluar...
    pause >nul
)
