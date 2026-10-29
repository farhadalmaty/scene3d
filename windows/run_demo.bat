@echo off
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
scene3d demo -o data\demo
scene3d plot data\demo\graph -o data\demo\plan.png
start "" data\demo\plan.png
