# MADDPG para Control de Flexibilidad Energética en Comunidades Interactivas

## Tema de Tesis
**MULTI-AGENTE DE APRENDIZAJE PROFUNDO POR REFUERZO PARA EL CONTROL DE LA FLEXIBILIDAD ENERGÉTICA EN COMUNIDADES INTERACTIVAS CON LA RED ELÉCTRICA PÚBLICA**

Implementación de Multi-Agent Deep Deterministic Policy Gradient (MADDPG) aplicado al control coordinado de edificios inteligentes en comunidades energéticas que interactúan con la red eléctrica pública. El sistema permite gestionar de manera óptima la flexibilidad energética mediante agentes autónomos que aprenden a coordinar consumo, almacenamiento y generación distribuida para maximizar eficiencia y minimizar costos.

## Instalación

### 1. Crear entorno virtual (Python 3.11.9 recomendado)

```bash
# Windows PowerShell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias principales

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Instalar CityLearn v2 (instalación manual requerida)

CityLearn se instala de forma independiente debido a conflictos de dependencias:

```bash
# Instalar CityLearn sin dependencias automáticas
pip install citylearn==2.5.0 --no-deps

# Instalar dependencias compatibles manualmente
pip install gymnasium==0.28.1 pandas "scikit-learn<=1.2.2" simplejson torchvision
```

**Nota**: Las dependencias `doe-xstock`, `nrel-pysam` y `openstudio` no se instalan porque requieren OpenStudio que no está disponible para Python 3.11 en Windows. El proyecto funciona sin ellas usando esquemas básicos de CityLearn.

## Uso

### Entrenamiento

```bash
cd maddpg_citylearn
python -m maddpg_tesis.scripts.train_citylearn
```

### API de Inferencia

```bash
uvicorn maddpg_tesis.api.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t maddpg-citylearn .
docker run -p 8000:8000 -v $(pwd)/models:/app/models maddpg-citylearn
```

## Objetivo Principal

**MULTI-AGENTE DE APRENDIZAJE PROFUNDO POR REFUERZO PARA EL CONTROL DE LA FLEXIBILIDAD ENERGÉTICA EN COMUNIDADES INTERACTIVAS CON LA RED ELÉCTRICA**

El sistema MADDPG entrena agentes autónomos (uno por edificio) que aprenden políticas coordinadas para:

- 📉 **Peak Shaving**: Reducir picos de demanda agregada de la comunidad
- ⚡ **Valley Filling**: Desplazar consumo a horas de baja demanda
- 🔋 **Self-Consumption**: Maximizar uso de generación solar local
- 💰 **Cost Optimization**: Responder a señales de precio dinámico
- 🌱 **Reducción de CO₂**: Minimizar emisiones asociadas al consumo

### Recursos Controlables por Agente

| Recurso | Acción del Agente |
|---------|-------------------|
| Batería estacionaria | Carga/descarga |
| Vehículo Eléctrico (EV) | Carga diferible |
| HVAC | Setpoints temperatura |
| DHW (Agua caliente) | Scheduling |

### Función de Recompensa Personalizable

La recompensa pondera múltiples objetivos de flexibilidad:

```yaml
reward_weights:
  cost: 1.0       # Penaliza costo energético
  peak: 0.5       # Penaliza picos de demanda
  co2: 0.3        # Penaliza emisiones
  discomfort: 0.2 # Penaliza disconfort térmico
```

## Alineación con "Guía Integral 2025 para Despliegue de Modelos ML/DL/LLM"

| Numeral | Tema | Implementación |
|---------|------|----------------|
| **1. Introducción** | Contexto del despliegue | `README.md`, `THESIS_CONTEXT.md`, `DEPLOYMENT_GUIDE.md` |
| **2. Contenedorización Docker** | | |
| 2.1 Buenas prácticas | Multi-stage, slim, no-root | `Dockerfile` con `python:3.11-slim`, `appuser:1001` |
| 2.2 Manejo de pesos | Volúmenes para modelos | PVC en `kubernetes/configmap-pvc.yaml` |
| **3. Orquestación Kubernetes** | | |
| 3.1 Componentes clave | Deployments, HPA | `deployment.yaml`, `hpa.yaml` (2-10 pods) |
| 3.2 Asignación GPU | nodeSelector, taints | ⚠️ No requerido (CPU-only) |
| 3.3 Frameworks serving | KServe, Ray Serve | FastAPI directo (suficiente para MADDPG) |
| **4. Despliegue ML** | Estrategias | FastAPI + Kubernetes + Docker |
| **5. Despliegue DL** | | |
| 5.1 Optimización | Cuantización | ⚠️ PyTorch nativo (optimización futura) |
| 5.2 Frameworks inferencia | TorchServe, Triton | FastAPI (modelo <50MB) |
| **6. Despliegue LLM** | Motores, técnicas | ➖ N/A (MADDPG no es LLM) |
| **7. Criterios Seguridad** | | |
| 7.1 Contenedores | Trivy, no-root | CI/CD con Trivy, `USER appuser` |
| 7.2 Infraestructura | NetworkPolicy, limits | `networkpolicy.yaml`, resource limits |
| 7.3 LLM | Prompt injection | ➖ N/A |
| **8. Monitoreo** | Observabilidad | `/health`, `/ready`, `/metrics` endpoints |
| **9. Lista Verificación** | Checklist | Ver `DEPLOYMENT_GUIDE.md` |

### Archivos Clave de Despliegue

```
kubernetes/
├── deployment.yaml      # Pods con security context, probes
├── service.yaml         # ClusterIP + LoadBalancer
├── hpa.yaml             # Auto-scaling CPU/memoria
├── configmap-pvc.yaml   # Configuración externalizada
├── ingress.yaml         # Exposición externa nginx
└── networkpolicy.yaml   # Políticas de red (seguridad)
```

### Utilidades adicionales

- Validar dataset y entorno: `python scripts/validate_dataset.py`.
- Probar API localmente: `uvicorn maddpg_tesis.api.main:app --reload` y `/docs`.
- Ejecutar tests rápidos: `pytest -q` (se salta carga de checkpoint con `SKIP_MODEL_LOAD_FOR_TESTS=1`).
- **Plan completo de despliegue**: Ver `DEPLOYMENT_GUIDE.md`
