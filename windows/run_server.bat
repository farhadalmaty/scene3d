@echo off
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
echo API docs will be at http://127.0.0.1:8000/docs
scene3d serve --data data --host 127.0.0.1 --port 8000
