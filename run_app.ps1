# Starts the SpaceXperiment control panel and opens it in your browser.
#   .\run_app.ps1                 (if scripts are blocked:  powershell -ExecutionPolicy Bypass -File .\run_app.ps1)
param([int]$Port = 8767)
Set-Location $PSScriptRoot
python src\app\server.py --open --port $Port
