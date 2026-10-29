@echo off
rem Usage: windows\annotate_video.bat [scene_folder] [only_tags]   e.g.  windows\annotate_video.bat data\scene1 mug,book
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
set PYTHONUTF8=1
set SC=%~1
if "%SC%"=="" set SC=data\scene1
scene3d annotate "%SC%" --only "%~2"
if exist "%SC%\annotated.mp4" start "" "%SC%\annotated.mp4"
