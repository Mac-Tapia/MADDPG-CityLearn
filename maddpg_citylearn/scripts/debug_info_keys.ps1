# PowerShell helper para ejecutar debug_info_keys.py con PYTHONPATH correcto
$env:PYTHONPATH = "$PSScriptRoot/../src"
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$python = Join-Path $root ".venv\Scripts\python.exe"
& $python "$PSScriptRoot/debug_info_keys.py"
