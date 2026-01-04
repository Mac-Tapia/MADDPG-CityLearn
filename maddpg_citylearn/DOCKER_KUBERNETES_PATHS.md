# Rutas de Docker y Kubernetes - MADDPG CityLearn

## 📦 Docker - Rutas y Configuración

### Dockerfile Principal
**Ubicación**: `maddpg_citylearn/Dockerfile`

**Ruta Absoluta**: `/home/runner/work/MADDPG-CityLearn/MADDPG-CityLearn/maddpg_citylearn/Dockerfile`

**Puntos Clave del Dockerfile**:
```dockerfile
# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Usuario no-root para seguridad
USER appuser (UID: 1001)

# Rutas de la aplicación en el contenedor:
/app/src/              # Código fuente (PYTHONPATH=/app/src)
/app/models/           # Modelos entrenados (montado como volumen)
/app/configs/          # Archivos de configuración
/app/static/           # Archivos estáticos
/app/logs/             # Logs de la aplicación

# Puerto expuesto
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "maddpg_tesis.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
**Ubicación**: `maddpg_citylearn/docker-compose.yml`

**Ruta Absoluta**: `/home/runner/work/MADDPG-CityLearn/MADDPG-CityLearn/maddpg_citylearn/docker-compose.yml`

**Configuración del Servicio**:
```yaml
servicios:
  maddpg-api:
    container_name: maddpg-citylearn-api
    ports:
      - "8000:8000"  # Host:Container
    
    # Volúmenes montados (Host -> Container):
    volumes:
      - ./models:/app/models:ro          # Modelos (read-only)
      - ./configs:/app/configs:ro        # Configuración (read-only)
      - ./logs:/app/logs                 # Logs (read-write)
    
    # Red de Docker
    networks:
      - maddpg-network
```

**Cómo ejecutar Docker Compose**:
```bash
# Desde el directorio maddpg_citylearn/
cd /home/runner/work/MADDPG-CityLearn/MADDPG-CityLearn/maddpg_citylearn

# Construir y ejecutar
docker-compose up -d

# Ver logs
docker-compose logs -f

# La API estará disponible en:
# http://localhost:8000
# http://localhost:8000/docs (Swagger UI)
```

---

## ☸️ Kubernetes - Rutas y Configuración

### Archivos de Configuración Kubernetes
**Ubicación Base**: `maddpg_citylearn/kubernetes/`

**Ruta Absoluta**: `/home/runner/work/MADDPG-CityLearn/MADDPG-CityLearn/maddpg_citylearn/kubernetes/`

```
kubernetes/
├── deployment.yaml              # Despliegue principal (2 réplicas)
├── deployment-gpu.yaml          # Despliegue con GPU NVIDIA
├── deployment-local.yaml        # Para Docker Desktop/Minikube
├── deployment-secure.yaml       # Con políticas de seguridad avanzadas
├── service.yaml                 # Servicios ClusterIP + LoadBalancer
├── hpa.yaml                     # Horizontal Pod Autoscaler (2-10 pods)
├── configmap-pvc.yaml          # ConfigMap + PersistentVolumeClaim
├── ingress.yaml                 # Exposición externa con nginx
├── networkpolicy.yaml           # Políticas de red
├── monitoring.yaml              # Prometheus ServiceMonitor
├── secrets.yaml                 # Gestión de secretos
└── rbac.yaml                    # Control de accesos
```

### Deployment Principal (deployment.yaml)

**Namespace**: `default`
**Nombre del Deployment**: `maddpg-citylearn`
**Réplicas**: 2 pods

**Rutas dentro de los Pods Kubernetes**:
```yaml
# Contenedor principal
Container: maddpg-api
  Imagen: maddpg-citylearn:latest
  Puerto: 8000
  
  # Variables de entorno
  PYTHONPATH: /app/src
  API_HOST: 0.0.0.0
  API_PORT: 8000
  LOG_LEVEL: INFO
  
  # Montajes de volúmenes:
  /app/models  -> PVC: maddpg-models-pvc (read-only)
  /app/configs -> ConfigMap: maddpg-config (read-only)
  /app/logs    -> EmptyDir (temporal)
  
  # Probes (health checks):
  Liveness:  GET /health (port 8000)
  Readiness: GET /health (port 8000)
```

### Services (service.yaml)

**1. ClusterIP Service** (interno):
```yaml
Nombre: maddpg-citylearn-service
Tipo: ClusterIP
Puerto: 80 -> 8000
Acceso interno: http://maddpg-citylearn-service.default.svc.cluster.local
```

**2. LoadBalancer Service** (externo):
```yaml
Nombre: maddpg-citylearn-lb
Tipo: LoadBalancer
Puerto: 8000 -> 8000
Acceso externo: http://<EXTERNAL-IP>:8000
```

### Cómo desplegar en Kubernetes

**Opción 1: Despliegue completo**
```bash
cd /home/runner/work/MADDPG-CityLearn/MADDPG-CityLearn/maddpg_citylearn/kubernetes

# 1. Crear ConfigMap y PVC
kubectl apply -f configmap-pvc.yaml

# 2. Desplegar la aplicación
kubectl apply -f deployment.yaml

# 3. Crear servicios
kubectl apply -f service.yaml

# 4. (Opcional) Auto-scaling
kubectl apply -f hpa.yaml

# 5. (Opcional) Ingress para acceso externo
kubectl apply -f ingress.yaml
```

**Opción 2: Usando Kustomize**
```bash
cd /home/runner/work/MADDPG-CityLearn/MADDPG-CityLearn/maddpg_citylearn/kubernetes

# Despliegue base
kubectl apply -k .

# Despliegue con GPU
kubectl apply -k overlays/gpu/

# Despliegue de producción
kubectl apply -k overlays/production/
```

### Verificar el despliegue

```bash
# Ver pods en ejecución
kubectl get pods -l app=maddpg-citylearn

# Ver servicios
kubectl get svc -l app=maddpg-citylearn

# Ver logs de un pod
kubectl logs -f <pod-name>

# Acceder a la shell de un pod
kubectl exec -it <pod-name> -- /bin/bash

# Port-forward para acceso local
kubectl port-forward svc/maddpg-citylearn-service 8000:80
# Luego acceder: http://localhost:8000
```

---

## 🗺️ Mapa Completo de Rutas

### En el Sistema de Archivos (Host)
```
/home/runner/work/MADDPG-CityLearn/MADDPG-CityLearn/
└── maddpg_citylearn/
    ├── Dockerfile                    # Configuración Docker
    ├── docker-compose.yml            # Orquestación Docker
    ├── kubernetes/                   # Manifiestos Kubernetes
    │   ├── deployment.yaml
    │   ├── service.yaml
    │   └── ...
    ├── models/                       # Modelos entrenados (HOST)
    ├── configs/                      # Configuración (HOST)
    ├── src/                         # Código fuente
    └── logs/                        # Logs de aplicación
```

### En Contenedor Docker
```
/app/
├── src/                  # Código (copiado en build)
├── models/               # Montado desde ./models (volumen)
├── configs/              # Montado desde ./configs (volumen)
├── static/               # Archivos estáticos (copiado en build)
└── logs/                 # Montado desde ./logs (volumen)
```

### En Pod de Kubernetes
```
/app/
├── src/                  # Código (en la imagen)
├── models/               # PVC: maddpg-models-pvc
├── configs/              # ConfigMap: maddpg-config
├── static/               # En la imagen
└── logs/                 # EmptyDir (temporal por pod)
```

---

## 🌐 Endpoints de la API

Una vez desplegado (Docker o Kubernetes), la API está disponible en:

```
http://localhost:8000                    # Raíz
http://localhost:8000/health            # Health check
http://localhost:8000/ready             # Readiness check
http://localhost:8000/metrics           # Métricas Prometheus
http://localhost:8000/predict           # Predicción (POST)
http://localhost:8000/docs              # Documentación Swagger
http://localhost:8000/redoc             # Documentación ReDoc
```

---

## 📊 Resumen de Ubicaciones

| Componente | Ubicación en Repositorio | Ubicación en Container/Pod |
|-----------|-------------------------|---------------------------|
| **Dockerfile** | `maddpg_citylearn/Dockerfile` | N/A |
| **Docker Compose** | `maddpg_citylearn/docker-compose.yml` | N/A |
| **K8s Manifests** | `maddpg_citylearn/kubernetes/*.yaml` | N/A |
| **Código Fuente** | `maddpg_citylearn/src/` | `/app/src/` |
| **Modelos** | `maddpg_citylearn/models/` | `/app/models/` |
| **Configs** | `maddpg_citylearn/configs/` | `/app/configs/` |
| **Logs** | `maddpg_citylearn/logs/` | `/app/logs/` |
| **API Port** | N/A | `8000` |

---

## 🚀 Arquitectura de Despliegue

```
┌─────────────────────────────────────────────────────────────────┐
│                    REPOSITORIO (Host)                            │
│  /home/runner/work/MADDPG-CityLearn/MADDPG-CityLearn/          │
│  └── maddpg_citylearn/                                          │
│      ├── Dockerfile ──────────────┐                             │
│      ├── docker-compose.yml       │                             │
│      ├── kubernetes/              │                             │
│      ├── src/                     │                             │
│      ├── models/                  │                             │
│      └── configs/                 │                             │
└───────────────────────────────────┼─────────────────────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │   DOCKER BUILD                 │
                    │   Image: maddpg-citylearn      │
                    └───────────┬────────────────────┘
                                │
                ┌───────────────┴────────────────┐
                │                                │
        ┌───────▼─────────┐           ┌────────▼──────────┐
        │  DOCKER RUN     │           │   KUBERNETES      │
        │  Container:     │           │   Deployment:     │
        │  maddpg-api     │           │   maddpg-citylearn│
        │                 │           │                   │
        │  /app/          │           │   Pods: 2-10      │
        │  ├── src/       │           │   ┌─────────────┐ │
        │  ├── models/    │           │   │ Pod 1       │ │
        │  ├── configs/   │           │   │ /app/       │ │
        │  └── logs/      │           │   │ ├── src/    │ │
        │                 │           │   │ ├── models/ │ │
        │  Port: 8000     │           │   │ └── configs/│ │
        └─────────────────┘           │   └─────────────┘ │
                                      │   ┌─────────────┐ │
                                      │   │ Pod 2       │ │
                                      │   │ /app/       │ │
                                      │   └─────────────┘ │
                                      │                   │
                                      │   Service:        │
                                      │   Port: 80->8000  │
                                      └───────────────────┘
```

---

## 📝 Notas Adicionales

1. **Docker**: Ideal para desarrollo local y testing
   - Ejecución simple con `docker-compose up`
   - Un solo contenedor
   - Volúmenes directos al filesystem

2. **Kubernetes**: Para producción y escalabilidad
   - Múltiples replicas (2-10 pods)
   - Auto-scaling con HPA
   - Persistencia con PVC
   - Load balancing automático
   - Health checks y auto-recovery

3. **Seguridad**: Ambas implementaciones usan:
   - Usuario no-root (appuser:1001)
   - Security contexts
   - Read-only volumes donde aplica
   - Network policies (K8s)

4. **Monitoreo**: 
   - Health endpoint: `/health`
   - Metrics endpoint: `/metrics` (Prometheus format)
   - Logs centralizados en `/app/logs`
