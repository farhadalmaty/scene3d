@echo off
rem Usage: windows\process_video.bat VIDEO [frames] [camera]   camera = main ^| ultrawide ^| tele   (0.5x = ultrawide)
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
if "%~1"=="" (echo Usage: process_video.bat PATH_TO_VIDEO [frames] [camera] ^& exit /b 1)
set FR=%~2
if "%FR%"=="" set FR=24
set CAM=
if not "%~3"=="" set CAM=--camera %~3
scene3d process "%~1" -o data\scene1 --frames %FR% --plot %CAM%
if exist data\scene1\plan.png start "" data\scene1\plan.png
