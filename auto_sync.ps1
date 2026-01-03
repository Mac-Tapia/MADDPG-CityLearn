# auto_sync.ps1 - Sincronización automática con GitHub
# Uso: .\auto_sync.ps1 o programar en Task Scheduler

param(
    [string]$Message = "Auto-sync: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
)

$ErrorActionPreference = "Stop"
$RepoPath = "D:\PROJECIA\MADDPG_TAREA"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Auto-Sync Repository" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Ir al directorio del repositorio
Set-Location $RepoPath

# Verificar si hay cambios
$status = git status --porcelain
if (-not $status) {
    Write-Host "[INFO] No hay cambios para sincronizar" -ForegroundColor Yellow
    exit 0
}

Write-Host "[INFO] Cambios detectados:" -ForegroundColor Green
git status --short

# Agregar todos los cambios
Write-Host ""
Write-Host "[INFO] Agregando cambios..." -ForegroundColor Green
git add -A

# Commit
Write-Host "[INFO] Creando commit..." -ForegroundColor Green
git commit -m $Message

# Push
Write-Host "[INFO] Subiendo a GitHub..." -ForegroundColor Green
git push origin main

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Sincronización completada!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
