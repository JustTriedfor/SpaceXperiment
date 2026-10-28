# Runs the unit tests from the terminal.
#   .\run_tests.ps1
Set-Location $PSScriptRoot
python -m pytest -q
