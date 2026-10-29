@echo off
rem Usage: windows\compress_video.bat INPUT_VIDEO [OUTPUT.mp4]   (default output: <input>_small.mp4)
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
if "%~1"=="" (echo Usage: compress_video.bat INPUT_VIDEO [OUTPUT.mp4] ^& exit /b 1)
if "%~2"=="" (scene3d compress "%~1") else (scene3d compress "%~1" -o "%~2")
