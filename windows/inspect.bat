@echo off
rem Usage: windows\inspect.bat [scene_folder] [ids]   e.g.  windows\inspect.bat data\scene1 5,112,140
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
set SC=%~1
if "%SC%"=="" set SC=data\scene1
scene3d inspect "%SC%" --ids "%~2"
if exist "%SC%\inspect" start "" "%SC%\inspect"
