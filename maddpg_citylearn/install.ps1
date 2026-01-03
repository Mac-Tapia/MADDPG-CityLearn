# Script de instalación automática para MADDPG CityLearn
# Ejecutar desde el directorio maddpg_citylearn/

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Instalación MADDPG CityLearn" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar Python 3.11
Write-Host "Verificando Python 3.11..." -ForegroundColor Yellow
$pythonVersion = & py -3.11 --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python 3.11 no está instalado" -ForegroundColor Red
    Write-Host "Instala Python 3.11.9 desde https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
Write-Host "✓ $pythonVersion encontrado" -ForegroundColor Green
Write-Host ""

# Paso 1: Actualizar pip
Write-Host "Paso 1: Actualizando pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: No se pudo actualizar pip" -ForegroundColor Red
    exit 1
}
Write-Host "✓ pip actualizado" -ForegroundColor Green
Write-Host ""

# Paso 2: Instalar dependencias principales
Write-Host "Paso 2: Instalando dependencias principales..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Falló la instalación de requirements.txt" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Dependencias principales instaladas" -ForegroundColor Green
Write-Host ""

# Paso 3: Instalar CityLearn de forma independiente
Write-Host "Paso 3: Instalando CityLearn v2 (independiente)..." -ForegroundColor Yellow
pip install citylearn==2.5.0 --no-deps
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Falló la instalación de CityLearn" -ForegroundColor Red
    exit 1
}
Write-Host "✓ CityLearn 2.5.0 instalado" -ForegroundColor Green
Write-Host ""

# Paso 4: Instalar dependencias de CityLearn
Write-Host "Paso 4: Instalando dependencias de CityLearn..." -ForegroundColor Yellow
pip install gymnasium==0.28.1 pandas "scikit-learn<=1.2.2" simplejson torchvision
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Falló la instalación de dependencias de CityLearn" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Dependencias de CityLearn instaladas" -ForegroundColor Green
Write-Host ""

# Verificación final
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Verificación de instalación" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$packages = @("torch", "numpy", "fastapi", "citylearn", "gymnasium", "pydantic")
foreach ($pkg in $packages) {
    $version = pip show $pkg 2>$null | Select-String "Version:"
    if ($version) {
        Write-Host "✓ $pkg instalado - $version" -ForegroundColor Green
    } else {
        Write-Host "✗ $pkg NO instalado" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Instalación completada exitosamente" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Ahora puedes ejecutar:" -ForegroundColor Yellow
Write-Host "  - Entrenamiento: python -m maddpg_tesis.scripts.train_citylearn" -ForegroundColor White
Write-Host "  - API: uvicorn maddpg_tesis.api.main:app --reload" -ForegroundColor White
Write-Host ""
