@echo off
REM OrthoRoute Plugin Launcher
REM Quick launcher for the OrthoRoute PCB autorouting plugin

echo.
echo ======================================
echo   OrthoRoute PCB Autorouting Plugin
echo ======================================
echo.

cd /d "%~dp0src"
python orthoroute_plugin.py

pause
