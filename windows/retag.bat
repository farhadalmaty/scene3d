@echo off
rem Usage: windows\retag.bat [scene_folder] [--compare]  ; rename objects from saved crops (no depth model)
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
set SC=%~1
if "%SC%"=="" set SC=data\scene1
scene3d retag "%SC%" %2 %3 %4
