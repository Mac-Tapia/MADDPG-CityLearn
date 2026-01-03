# =============================================================================
# Script de Instalación y Configuración de Minikube
# =============================================================================

$ErrorActionPreference = "Stop"

function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Cyan }
function Write-Success { Write-Host "[OK] $args" -ForegroundColor Green }
function Write-Error { Write-Host "[ERROR] $args" -ForegroundColor Red }

Write-Info "Instalando Minikube con Chocolatey..."

# Verificar si Chocolatey está instalado
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Info "Chocolatey no está instalado. Instalando..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}

# Instalar Minikube
Write-Info "Instalando Minikube..."
choco install minikube -y

# Iniciar Minikube con Docker driver
Write-Info "Iniciando Minikube con driver Docker..."
minikube start --driver=docker --cpus=4 --memory=8192

# Verificar estado
Write-Info "Verificando estado de Minikube..."
minikube status

# Configurar kubectl para usar Minikube
kubectl config use-context minikube

# Habilitar addons útiles
Write-Info "Habilitando addons..."
minikube addons enable metrics-server
minikube addons enable dashboard
minikube addons enable ingress

Write-Success "Minikube instalado y configurado correctamente!"
Write-Info "Puedes abrir el dashboard con: minikube dashboard"
