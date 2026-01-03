@echo off
REM auto_sync.bat - Sincronización automática con GitHub
REM Uso: auto_sync.bat o programar en Task Scheduler

cd /d D:\PROJECIA\MADDPG_TAREA

echo ========================================
echo   Auto-Sync Repository
echo ========================================

REM Verificar si hay cambios
git status --porcelain > nul 2>&1
git diff --quiet HEAD
if %ERRORLEVEL% EQU 0 (
    echo [INFO] No hay cambios para sincronizar
    exit /b 0
)

echo [INFO] Cambios detectados:
git status --short

echo.
echo [INFO] Agregando cambios...
git add -A

echo [INFO] Creando commit...
git commit -m "Auto-sync: %date% %time%"

echo [INFO] Subiendo a GitHub...
git push origin main

echo.
echo ========================================
echo   Sincronizacion completada!
echo ========================================
