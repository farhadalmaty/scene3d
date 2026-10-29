@echo off
rem One-time setup on Windows 10/11. Run from anywhere: double-click or `windows\setup.bat`.
cd /d "%~dp0\.."
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher "py" not found. Install 64-bit Python 3.11 from python.org and tick "Add python.exe to PATH".
  exit /b 1
)
if not exist .venv (
  rem Depth Anything 3 needs Python 3.9-3.13; torch wheels are safest on 3.10-3.12. Never fall back to "latest".
  py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || py -3.10 -m venv .venv 2>nul || py -3.13 -m venv .venv 2>nul
)
if not exist .venv\Scripts\activate.bat (
  echo No suitable Python found. Install 64-bit Python 3.12 from python.org ^(tick "Add python.exe to PATH"^) and rerun.
  exit /b 1
)
call .venv\Scripts\activate.bat
python -c "import sys; print('Python', sys.version.split()[0]); sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)"
if errorlevel 1 (
  echo The .venv uses an unsupported Python ^(need 3.10-3.13^). Delete the .venv folder, install Python 3.12 and rerun.
  exit /b 1
)
python -m pip install --upgrade pip
echo === PyTorch (CPU build) ===
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 exit /b 1
echo === scene3d + FastSAM + CLIP + server + agent ===
pip install -e ".[all]"
if errorlevel 1 exit /b 1
echo === Depth Anything 3 (camera pose + depth from video) ===
where git >nul 2>nul
if errorlevel 1 (
  echo git not found. Install Git for Windows, or download the repo ZIP from
  echo https://github.com/ByteDance-Seed/depth-anything-3 and unpack it to third_party\depth-anything-3
  exit /b 1
)
if not exist third_party mkdir third_party
if not exist third_party\depth-anything-3 git clone https://github.com/ByteDance-Seed/depth-anything-3 third_party\depth-anything-3
pip install -e third_party\depth-anything-3
if errorlevel 1 (
  echo Full install failed ^(often xformers on Windows^). Retrying without dependencies...
  pip install -e third_party\depth-anything-3 --no-deps
  if errorlevel 1 (
    echo Depth Anything 3 could NOT be installed. Send the error text above.
    exit /b 1
  )
  echo If "scene3d process" later complains about a missing module, pip install that module.
)
echo.
echo Done. Check the install without any models:  windows\run_tests.bat   and   windows\run_demo.bat
