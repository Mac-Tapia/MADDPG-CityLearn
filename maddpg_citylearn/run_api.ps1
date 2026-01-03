# Script para ejecutar la API MADDPG localmente
# Uso: .\run_api.ps1

$ErrorActionPreference = "Stop"

# Configurar directorio de trabajo
Set-Location $PSScriptRoot

# Activar entorno virtual
$venvPath = "..\..\.venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    & $venvPath
    Write-Host "✓ Entorno virtual activado" -ForegroundColor Green
} else {
    Write-Host "⚠ Entorno virtual no encontrado en $venvPath" -ForegroundColor Yellow
}

# Configurar PYTHONPATH
$env:PYTHONPATH = "$PSScriptRoot\src"

# Verificar que el modelo existe
$modelPath = "models\citylearn_maddpg\maddpg.pt"
if (Test-Path $modelPath) {
    Write-Host "✓ Modelo encontrado: $modelPath" -ForegroundColor Green
} else {
    Write-Host "✗ Modelo no encontrado: $modelPath" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  Iniciando API MADDPG CityLearn" -ForegroundColor Cyan
Write-Host "  URL: http://localhost:8000" -ForegroundColor Cyan
Write-Host "  Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# Iniciar la API
python -m uvicorn maddpg_tesis.api.main:app --host 0.0.0.0 --port 8000
