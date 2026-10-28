@echo off
rem Double-click to start the SpaceXperiment control panel (opens your browser).
cd /d "%~dp0"
python src\app\server.py --open
pause
