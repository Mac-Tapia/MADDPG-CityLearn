Set-Location "$PSScriptRoot\.."
$projRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$env:PYTHONPATH = "$projRoot\maddpg_citylearn\src"
$python = "$projRoot\.venv\Scripts\python.exe"
if (-Not (Test-Path $python)) {
	Write-Error "No se encontró Python en $python"
	exit 1
}

& "$python" "$projRoot\maddpg_citylearn\scripts\train_citylearn.py"
