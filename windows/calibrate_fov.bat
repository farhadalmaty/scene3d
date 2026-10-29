@echo off
rem Usage: windows\calibrate_fov.bat VIDEO   - finds the camera field of view that fits the footage (e.g. for 0.5x ultra-wide)
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
if "%~1"=="" (echo Usage: calibrate_fov.bat VIDEO ^& exit /b 1)
scene3d calibrate-fov "%~1"
