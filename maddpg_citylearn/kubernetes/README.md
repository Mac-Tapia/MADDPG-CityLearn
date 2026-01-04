# Guía de Despliegue en Kubernetes - MADDPG CTDE CityLearn

## Descripción

Despliegue de **MADDPG CTDE Cooperative** (Multi-Agent Deep Deterministic Policy Gradient) para control de flexibilidad energética en comunidades de edificios inteligentes.

### Modelo Entrenado

| Parámetro | Valor |
|-----------|-------|
| **Paradigma** | CTDE (Centralized Training, Decentralized Execution) |
| **Agentes** | 17 edificios cooperativos |
| **Episodios** | 50 (438,000 steps totales) |
| **Dataset** | `citylearn_challenge_2022_phase_all_plus_evs` |
| **Recompensa** | Team Reward Cooperativo |
| **Observación** | (17, 28) - 28 features por agente |
| **Acciones** | (17, 6) - 6 acciones continuas en [-1, 1] |

### KPIs del Modelo (vs Baseline)

| Métrica | Valor |
|---------|-------|
| Cost Total | 1.185 |
| Carbon Emissions | 1.207 |
| Daily Peak Average | 1.028 |
| All Time Peak | 1.319 |

## Requisitos Previos

### 1. Habilitar Kubernetes en Docker Desktop

1. Abrir Docker Desktop
2. Ir a **Settings** (⚙️) → **Kubernetes**
3. Marcar **Enable Kubernetes**
4. Click en **Apply & Restart**
5. Esperar a que Kubernetes esté listo (icono verde)

### 2. Verificar la Instalación

```powershell
# Verificar kubectl
kubectl version --client

# Verificar conexión al cluster
kubectl cluster-info

# Ver nodos disponibles
kubectl get nodes
```

## Estructura de Manifiestos

```
kubernetes/
├── kustomization.yaml         # Base kustomize - CTDE Cooperative
├── configmap-pvc.yaml         # ConfigMaps con hiperparámetros CTDE
├── deployment.yaml            # Deployment principal (CPU)
├── deployment-gpu.yaml        # Deployment con GPU NVIDIA
├── service.yaml               # Services (ClusterIP + LoadBalancer)
├── hpa.yaml                   # Horizontal Pod Autoscaler
├── monitoring.yaml            # ServiceMonitor + PrometheusRule
├── ingress.yaml               # Ingress (nginx)
├── networkpolicy.yaml         # Políticas de red
├── deploy-k8s.ps1             # Script de despliegue
└── overlays/
    ├── gpu/                   # Overlay para GPU NVIDIA
    │   └── kustomization.yaml
    └── production/            # Overlay para producción
        └── kustomization.yaml
```

## Despliegue

### Opción 1: Script de Despliegue (Recomendado)

```powershell
cd maddpg_citylearn/kubernetes

# Despliegue completo
.\deploy-k8s.ps1

# Ver estado
.\deploy-k8s.ps1 -Action status

# Ver logs
.\deploy-k8s.ps1 -Action logs

# Probar endpoints
.\deploy-k8s.ps1 -Action test

# Port-forward para pruebas locales
.\deploy-k8s.ps1 -Action port-forward

# Eliminar recursos
.\deploy-k8s.ps1 -Action delete
```

### Opción 2: Kustomize

```powershell
# Despliegue base (CPU)
kubectl apply -k kubernetes/

# Despliegue con GPU
kubectl apply -k kubernetes/overlays/gpu/

# Despliegue producción
kubectl apply -k kubernetes/overlays/production/

# Ver manifiestos generados (sin aplicar)
kubectl kustomize kubernetes/
```

### Opción 3: Manifiestos Individuales

```powershell
# Aplicar en orden
kubectl apply -f kubernetes/configmap-pvc.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml
kubectl apply -f kubernetes/networkpolicy.yaml
kubectl apply -f kubernetes/ingress.yaml
```

## Verificación del Despliegue

```powershell
# Ver pods
kubectl get pods -l app=maddpg-citylearn -w

# Ver logs
kubectl logs -l app=maddpg-citylearn -f

# Ver servicios
kubectl get svc

# Ver HPA
kubectl get hpa

# Describir deployment
kubectl describe deployment maddpg-citylearn
```

## Acceso a la API

### Port-Forward (Desarrollo)

```powershell
kubectl port-forward svc/maddpg-citylearn-service 8000:80

# Probar
curl http://localhost:8000/health
```

### LoadBalancer (Docker Desktop)

```powershell
# El servicio LoadBalancer expone en localhost:8000
curl http://localhost:8000/health
```

### Ingress (Producción)

1. Instalar Ingress Controller:
```powershell
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml
```

2. Configurar DNS o `/etc/hosts`:
```
127.0.0.1 maddpg-citylearn.example.com
```

3. Acceder:
```powershell
curl http://maddpg-citylearn.example.com/health
```

## Monitoreo

### Métricas de Recursos

```powershell
# Uso de recursos por pod
kubectl top pods -l app=maddpg-citylearn

# Uso de recursos por nodo
kubectl top nodes
```

### Logs Centralizados

```powershell
# Logs de todos los pods
kubectl logs -l app=maddpg-citylearn --all-containers

# Logs con timestamps
kubectl logs -l app=maddpg-citylearn --timestamps
```

### Health Checks

```powershell
# Ver estado de probes
kubectl describe pod -l app=maddpg-citylearn | grep -A5 "Liveness\|Readiness"
```

## Escalado

### Manual

```powershell
# Escalar a 5 réplicas
kubectl scale deployment maddpg-citylearn --replicas=5
```

### Automático (HPA)

El HPA está configurado para:
- **Mínimo**: 2 réplicas
- **Máximo**: 10 réplicas
- **CPU target**: 70%
- **Memory target**: 80%

```powershell
# Ver estado del HPA
kubectl get hpa maddpg-citylearn-hpa -w
```

## Despliegue con GPU (NVIDIA)

### Requisitos

1. NVIDIA GPU en el nodo
2. NVIDIA Container Toolkit instalado
3. NVIDIA Device Plugin para Kubernetes

### Instalar Device Plugin

```powershell
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
```

### Desplegar con GPU

```powershell
kubectl apply -k kubernetes/overlays/gpu/
```

### Verificar GPU

```powershell
# Ver GPUs disponibles
kubectl describe node | grep -A5 "nvidia.com/gpu"

# Ver pods usando GPU
kubectl get pods -o json | jq '.items[].spec.containers[].resources'
```

## Troubleshooting

### Pod no inicia

```powershell
# Ver eventos
kubectl describe pod <pod-name>

# Ver logs del pod anterior
kubectl logs <pod-name> --previous
```

### Imagen no encontrada

```powershell
# Verificar imagen local
docker images | grep maddpg-citylearn

# El deployment usa imagePullPolicy: IfNotPresent
# Para forzar re-pull, cambiar a Always temporalmente
```

### Problemas de red

```powershell
# Verificar NetworkPolicy
kubectl describe networkpolicy maddpg-citylearn-network-policy

# Probar conectividad interna
kubectl run test-pod --rm -it --image=busybox -- wget -qO- http://maddpg-citylearn-service/health
```

### HPA no escala

```powershell
# Verificar métricas disponibles
kubectl top pods

# Si no hay métricas, instalar metrics-server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

## Limpieza

```powershell
# Eliminar todos los recursos
kubectl delete -k kubernetes/

# O usando el script
.\deploy-k8s.ps1 -Action delete
```

## Producción Checklist

- [ ] Cambiar `imagePullPolicy` a `Always`
- [ ] Configurar registry privado para la imagen
- [ ] Habilitar TLS en Ingress
- [ ] Configurar cert-manager para certificados
- [ ] Ajustar recursos según carga esperada
- [ ] Configurar backup de PVCs
- [ ] Implementar logging centralizado (ELK/Loki)
- [ ] Configurar alertas (Prometheus/AlertManager)
- [ ] Revisar NetworkPolicies
- [ ] Habilitar PodDisruptionBudget
