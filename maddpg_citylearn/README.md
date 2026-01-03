# CooperativeMADDPG para Control de Flexibilidad Energética en Comunidades Inteligentes

## Tema de Tesis

**Sistema Multi-Agente de Aprendizaje Profundo por Refuerzo para la Optimización de la Flexibilidad Energética en Comunidades Interactivas de Redes Eléctricas Inteligentes**

Implementación de **Cooperative MADDPG** (Multi-Agent Deep Deterministic Policy Gradient) con paradigma **CTDE** (Centralized Training, Decentralized Execution) aplicado al control coordinado de edificios inteligentes en comunidades energéticas. El sistema utiliza:

- 🤝 **Team Reward**: Todos los agentes reciben la misma recompensa global basada en métricas del distrito
- 🧠 **Coordinación Explícita**: Módulos de Mean-Field + Attention para comunicación inter-agentes
- 📊 **17 Edificios**: Dataset CityLearn Challenge 2022 Phase All + EVs
- ⚡ **GPU Acelerada**: PyTorch 2.5.1 + CUDA 12.1 (RTX 4060)

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

### Entrenamiento Cooperativo (CTDE + Team Reward)

```bash
cd maddpg_citylearn
$env:PYTHONPATH="src"; python -u scripts/train_citylearn.py

# O usando el script cooperativo dedicado:
$env:PYTHONPATH="src"; python -u scripts/train_cooperative.py
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

**Sistema Multi-Agente de Aprendizaje Profundo por Refuerzo para la Optimización de la Flexibilidad Energética en Comunidades Interactivas de Redes Eléctricas Inteligentes**

### Paradigma CTDE (Centralized Training, Decentralized Execution)

```
┌─────────────────────────────────────────────────┐
│           ENTRENAMIENTO CENTRALIZADO            │
│  ┌─────────────────────────────────────────┐   │
│  │  COORDINADOR (Mean-Field + Attention)   │   │
│  └─────────────────────────────────────────┘   │
│                    ↓                            │
│  ┌──────┐ ┌──────┐ ┌──────┐      ┌──────┐    │
│  │Actor1│ │Actor2│ │Actor3│ ...  │Actor17│    │
│  └──────┘ └──────┘ └──────┘      └──────┘    │
│       ↓       ↓       ↓              ↓        │
│  ┌─────────────────────────────────────────┐   │
│  │     CRITIC CENTRALIZADO (Q-global)      │   │
│  └─────────────────────────────────────────┘   │
│       ↓                                        │
│  ┌─────────────────────────────────────────┐   │
│  │  TEAM REWARD (misma para todos)         │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

El sistema **CooperativeMADDPG** entrena 17 agentes autónomos (uno por edificio) con:

- 📉 **Peak Shaving**: Reducir picos de demanda agregada de la comunidad
- ⚡ **Valley Filling**: Desplazar consumo a horas de baja demanda
- 🔋 **Self-Consumption**: Maximizar uso de generación solar local
- 💰 **Cost Optimization**: Responder a señales de precio dinámico
- 🌱 **Reducción de CO₂**: Minimizar emisiones asociadas al consumo
- 🤝 **Coordinación**: Mecanismos de atención y mean-field entre agentes

### Recursos Controlables por Agente

| Recurso | Acción del Agente |
| ------- | ----------------- |
| Batería estacionaria | Carga/descarga |
| Vehículo Eléctrico (EV) | Carga diferible |
| HVAC | Setpoints temperatura |
| DHW (Agua caliente) | Scheduling |

### Team Reward (Recompensa Cooperativa)

Todos los agentes reciben la **misma recompensa global** basada en métricas del distrito:

```python
# reward_functions.py - Team Reward
def calculate_team_reward(env) -> List[float]:
    total_cost = sum(b.net_electricity_consumption_cost[-1] for b in buildings)
    total_emissions = sum(b.net_electricity_consumption_emission[-1] for b in buildings)
    global_ramping = abs(current_total - previous_total)
    load_factor = np.var(consumptions)
    
    team_reward = -(
        weights.cost * total_cost +
        weights.carbon * total_emissions +
        weights.ramping * global_ramping +
        weights.load_factor * load_factor
    )
    return [team_reward] * n_buildings  # MISMA para todos
```

### Métricas de Evaluación (5 KPIs)

| Métrica | Peso | Descripción |
|---------|------|-------------|
| **Cost** | 25% | Costo energético total del distrito |
| **Carbon** | 25% | Emisiones de CO₂ totales |
| **Ramping** | 20% | Cambios abruptos en demanda |
| **Load Factor** | 15% | Factor de carga (pico vs promedio) |
| **Electricity** | 15% | Consumo eléctrico total |

## Alineación con "Guía Integral 2025 para Despliegue de Modelos ML/DL/LLM"

| Numeral | Tema | Implementación |
| ------- | ---- | -------------- |
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

```text
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
