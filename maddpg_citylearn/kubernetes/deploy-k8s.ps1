# =============================================================================
# Script de Despliegue Kubernetes - MADDPG CityLearn
# =============================================================================
# Uso:
#   .\deploy-k8s.ps1                    # Despliegue completo
#   .\deploy-k8s.ps1 -Action status     # Ver estado
#   .\deploy-k8s.ps1 -Action delete     # Eliminar recursos
#   .\deploy-k8s.ps1 -Action logs       # Ver logs
#   .\deploy-k8s.ps1 -UseGPU            # Despliegue con GPU
# =============================================================================

param(
    [ValidateSet("deploy", "status", "delete", "logs", "test", "port-forward")]
    [string]$Action = "deploy",
    
    [switch]$UseGPU,
    
    [string]$Namespace = "default",
    
    [string]$ImageTag = "latest"
)

$ErrorActionPreference = "Stop"
$K8S_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$APP_NAME = "maddpg-citylearn"

# Colores para output
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Cyan }
function Write-Success { Write-Host "[OK] $args" -ForegroundColor Green }
function Write-Warning { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-Error { Write-Host "[ERROR] $args" -ForegroundColor Red }

# Verificar que kubectl está disponible
function Test-Kubectl {
    try {
        $null = kubectl version --client --short 2>&1
        return $true
    } catch {
        Write-Error "kubectl no está instalado o no está en PATH"
        return $false
    }
}

# Verificar conexión al cluster
function Test-ClusterConnection {
    try {
        $context = kubectl config current-context 2>&1
        Write-Info "Conectado al cluster: $context"
        return $true
    } catch {
        Write-Error "No hay conexión al cluster de Kubernetes"
        return $false
    }
}

# Desplegar recursos en Kubernetes
# Nota: Usamos Publish- que es un verbo aprobado de PowerShell
function Publish-Resources {
    <#
    .SYNOPSIS
    Despliega los recursos de MADDPG CityLearn en Kubernetes
    #>
    Write-Info "Desplegando MADDPG CityLearn en Kubernetes..."
    
    # Verificar que la imagen existe localmente
    $imageExists = docker images -q "${APP_NAME}:${ImageTag}" 2>$null
    if (-not $imageExists) {
        Write-Warning "La imagen ${APP_NAME}:${ImageTag} no existe localmente"
        Write-Info "Ejecute primero: docker build -t ${APP_NAME}:${ImageTag} ."
    }
    
    # Crear namespace si no existe
    $nsExists = kubectl get namespace $Namespace 2>$null
    if (-not $nsExists -and $Namespace -ne "default") {
        Write-Info "Creando namespace: $Namespace"
        kubectl create namespace $Namespace
    }
    
    # Usar kustomize si está disponible
    if (Test-Path "$K8S_DIR\kustomization.yaml") {
        Write-Info "Desplegando con Kustomize..."
        kubectl apply -k $K8S_DIR -n $Namespace
    } else {
        # Despliegue manual en orden
        Write-Info "Desplegando recursos individuales..."
        
        $resources = @(
            "configmap-pvc.yaml",
            "deployment.yaml",
            "service.yaml",
            "hpa.yaml",
            "networkpolicy.yaml",
            "ingress.yaml"
        )
        
        foreach ($resource in $resources) {
            $path = Join-Path $K8S_DIR $resource
            if (Test-Path $path) {
                Write-Info "Aplicando: $resource"
                kubectl apply -f $path -n $Namespace
            }
        }
    }
    
    Write-Success "Despliegue completado"
    
    # Esperar a que los pods estén listos
    Write-Info "Esperando a que los pods estén listos..."
    kubectl rollout status deployment/$APP_NAME -n $Namespace --timeout=300s
    
    Get-Status
}

# Obtener estado
function Get-Status {
    Write-Info "Estado del despliegue:"
    Write-Host ""
    
    Write-Host "=== PODS ===" -ForegroundColor Yellow
    kubectl get pods -l app=$APP_NAME -n $Namespace -o wide
    Write-Host ""
    
    Write-Host "=== SERVICES ===" -ForegroundColor Yellow
    kubectl get svc -l app=$APP_NAME -n $Namespace
    Write-Host ""
    
    Write-Host "=== HPA ===" -ForegroundColor Yellow
    kubectl get hpa -n $Namespace
    Write-Host ""
    
    Write-Host "=== INGRESS ===" -ForegroundColor Yellow
    kubectl get ingress -n $Namespace
    Write-Host ""
    
    # Mostrar endpoints
    Write-Host "=== ENDPOINTS ===" -ForegroundColor Yellow
    $svcIP = kubectl get svc ${APP_NAME}-service -n $Namespace -o jsonpath='{.spec.clusterIP}' 2>$null
    if ($svcIP) {
        Write-Host "ClusterIP: http://${svcIP}:80"
    }
    
    $lbIP = kubectl get svc ${APP_NAME}-lb -n $Namespace -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>$null
    if ($lbIP) {
        Write-Host "LoadBalancer: http://${lbIP}:8000"
    }
}

# Ver logs
function Get-Logs {
    Write-Info "Logs de los pods:"
    
    $pods = kubectl get pods -l app=$APP_NAME -n $Namespace -o jsonpath='{.items[*].metadata.name}'
    $podList = $pods -split ' '
    
    foreach ($pod in $podList) {
        if ($pod) {
            Write-Host "`n=== Logs de $pod ===" -ForegroundColor Yellow
            kubectl logs $pod -n $Namespace --tail=50
        }
    }
}

# Eliminar recursos
function Remove-Resources {
    Write-Warning "Eliminando recursos de MADDPG CityLearn..."
    
    $confirm = Read-Host "¿Está seguro? (s/N)"
    if ($confirm -ne "s" -and $confirm -ne "S") {
        Write-Info "Operación cancelada"
        return
    }
    
    if (Test-Path "$K8S_DIR\kustomization.yaml") {
        kubectl delete -k $K8S_DIR -n $Namespace
    } else {
        $resources = @(
            "ingress.yaml",
            "networkpolicy.yaml",
            "hpa.yaml",
            "service.yaml",
            "deployment.yaml",
            "configmap-pvc.yaml"
        )
        
        foreach ($resource in $resources) {
            $path = Join-Path $K8S_DIR $resource
            if (Test-Path $path) {
                kubectl delete -f $path -n $Namespace --ignore-not-found
            }
        }
    }
    
    Write-Success "Recursos eliminados"
}

# Port forward para pruebas locales
function Start-PortForward {
    Write-Info "Iniciando port-forward a localhost:8000..."
    Write-Info "Presione Ctrl+C para detener"
    
    $pod = kubectl get pods -l app=$APP_NAME -n $Namespace -o jsonpath='{.items[0].metadata.name}'
    if ($pod) {
        kubectl port-forward $pod 8000:8000 -n $Namespace
    } else {
        Write-Error "No se encontraron pods en ejecución"
    }
}

# Probar endpoints
function Test-Endpoints {
    Write-Info "Probando endpoints de la API..."
    
    # Intentar con port-forward en background
    $pod = kubectl get pods -l app=$APP_NAME -n $Namespace -o jsonpath='{.items[0].metadata.name}' 2>$null
    
    if (-not $pod) {
        Write-Error "No hay pods disponibles"
        return
    }
    
    Write-Info "Iniciando port-forward temporal..."
    $job = Start-Job -ScriptBlock {
        kubectl port-forward $using:pod 8001:8000 -n $using:Namespace 2>&1
    }
    
    Start-Sleep -Seconds 3
    
    try {
        Write-Host "`n=== Health Check ===" -ForegroundColor Yellow
        $health = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get
        Write-Host ($health | ConvertTo-Json)
        
        Write-Host "`n=== Ready Check ===" -ForegroundColor Yellow
        $ready = Invoke-RestMethod -Uri "http://localhost:8001/ready" -Method Get
        Write-Host ($ready | ConvertTo-Json)
        
        Write-Host "`n=== Metrics ===" -ForegroundColor Yellow
        $metrics = Invoke-RestMethod -Uri "http://localhost:8001/metrics" -Method Get
        Write-Host ($metrics | ConvertTo-Json)
        
        Write-Success "Todos los endpoints funcionan correctamente"
    } catch {
        Write-Error "Error al probar endpoints: $_"
    } finally {
        Stop-Job $job -ErrorAction SilentlyContinue
        Remove-Job $job -ErrorAction SilentlyContinue
    }
}

# Main
if (-not (Test-Kubectl)) { exit 1 }
if (-not (Test-ClusterConnection)) { exit 1 }

switch ($Action) {
    "deploy" { Publish-Resources }
    "status" { Get-Status }
    "delete" { Remove-Resources }
    "logs" { Get-Logs }
    "test" { Test-Endpoints }
    "port-forward" { Start-PortForward }
}
