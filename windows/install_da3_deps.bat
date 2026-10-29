@echo off
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
python windows\install_da3_deps.py
python -c "from depth_anything_3.api import DepthAnything3; print(\"DA3 import OK\")"
