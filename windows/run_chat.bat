@echo off
rem Needs ANTHROPIC_API_KEY (set it once with:  setx ANTHROPIC_API_KEY "sk-ant-..."  then open a NEW window)
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
if "%ANTHROPIC_API_KEY%"=="" (echo ANTHROPIC_API_KEY is not set & exit /b 1)
scene3d chat data\scene1\graph
