@echo off
rem Usage: windows\process_video.bat C:\scene3d_videos\room.mp4 [frames]
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
if "%~1"=="" (echo Usage: process_video.bat PATH_TO_VIDEO [frames] & exit /b 1)
set FR=%~2
if "%FR%"=="" set FR=24
scene3d process "%~1" -o data\scene1 --frames %FR% --plot
if exist data\scene1\plan.png start "" data\scene1\plan.png
