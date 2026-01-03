# Contexto de Investigación - Tesis

## Tema de Tesis

**MULTI-AGENTE DE APRENDIZAJE PROFUNDO POR REFUERZO PARA EL CONTROL DE LA FLEXIBILIDAD ENERGÉTICA EN COMUNIDADES INTERACTIVAS CON LA RED ELÉCTRICA PÚBLICA**

## Objetivo de Investigación

Desarrollar e implementar un sistema multi-agente basado en Deep Reinforcement Learning (específicamente MADDPG) para controlar de manera óptima y coordinada la flexibilidad energética en comunidades de edificios inteligentes que interactúan con la red eléctrica pública.

## Componentes Clave de la Investigación

### 1. Sistema Multi-Agente
- **Agentes Autónomos**: Cada edificio/unidad de la comunidad opera como un agente independiente
- **Coordinación Distribuida**: Los agentes aprenden a coordinar sus acciones para optimización global
- **Descentralización**: Ejecución local autónoma tras entrenamiento centralizado

### 2. Control de Flexibilidad Energética

La flexibilidad energética se refiere a la capacidad de los edificios de ajustar su consumo, almacenamiento y generación de energía en respuesta a:

- **Señales de Precio**: Respuesta a tarifas dinámicas de electricidad
- **Demanda de Red**: ReduccPión de picos, valley-filling, load shifting
- **Estabilidad de Red**: Servicios auxiliares, balance oferta-demanda
- **Recursos Locales**: Optimización de generación solar, baterías, cargas controlables

#### Recursos Controlables:
- **Cargas Térmicas**: HVAC, calefacción, refrigeración (mayor inercia térmica)
- **Almacenamiento**: Baterías eléctricas, almacenamiento térmico
- **Generación Distribuida**: Paneles solares, cogeneración
- **Cargas Diferibles**: Vehículos eléctricos, electrodomésticos programables

### 3. Comunidades Interactivas con la Red Pública

Las comunidades energéticas representan agregaciones de edificios que:

- **Interactúan Colectivamente**: Presentan demanda/oferta agregada a la red
- **Se Benefician Mutuamente**: Intercambio local de energía, reducción de costos
- **Contribuyen a la Red**: Estabilidad, reducción de inversiones en infraestructura
- **Responden a Señales**: Precio, frecuencia, demanda de respuesta (demand response)

#### Características de Interacción:
- Punto de acoplamiento común (PCC - Point of Common Coupling)
- Medición neta agregada (net metering)
- Contratos de compra/venta con utilities
- Participación en mercados de flexibilidad

### 4. Aprendizaje Profundo por Refuerzo Multi-Agente

**¿Por qué MADDPG?**

1. **Acciones Continuas**: Control fino de setpoints de temperatura, carga de baterías
2. **Multi-Agente**: Múltiples edificios operando simultáneamente
3. **Coordinación**: Crítico centralizado observa acciones de todos los agentes
4. **Escalabilidad**: Agrega nuevos edificios sin rediseño total
5. **Off-Policy**: Aprendizaje eficiente de datos históricos

**Componentes del Algoritmo:**
- **Actor Descentralizado**: Cada edificio tiene su propia política de control
- **Crítico Centralizado**: Evalúa valor de acciones conjuntas considerando todos los agentes
- **Target Networks**: Estabilidad en entrenamiento
- **Replay Buffer**: Aprendizaje de experiencias pasadas

## Caso de Uso: CityLearn Challenge 2022 Phase All + EVs

El entorno **CityLearn** con el dataset `citylearn_challenge_2022_phase_all_plus_evs` simula:

- 🏢 Múltiples edificios con características distintas (residencial, comercial)
- 🚗 **Vehículos Eléctricos (EVs)** como cargas controlables y diferibles
- ☀️ Generación solar fotovoltaica distribuida
- 🔋 Sistemas de almacenamiento de energía (baterías estacionarias)
- 🌡️ Cargas térmicas controlables (HVAC, DHW)
- 💰 Señales de precio de electricidad dinámico
- 📊 Emisiones de carbono de la red
- ⚡ Demanda de red agregada para gestión de picos

### Importancia de los Vehículos Eléctricos

Los EVs son particularmente importantes para la flexibilidad energética porque:
- **Gran capacidad de almacenamiento**: Baterías de 40-100 kWh
- **Conectividad predecible**: Patrones de llegada/salida en horarios típicos
- **Cargas diferibles**: Flexibilidad en ventanas de carga (8-12 horas)
- **Potencial V2G**: Vehicle-to-Grid para servicios auxiliares a la red

### Métricas de Evaluación

Las métricas típicas incluyen:

1. **Costo Energético**: Minimizar gasto total de electricidad
2. **Pico de Demanda**: Reducir demanda máxima (peak shaving)
3. **Ramping**: Suavizar cambios abruptos de demanda
4. **Factor de Carga**: Mejorar utilización promedio vs pico
5. **Emisiones de CO₂**: Reducir huella de carbono
6. **Confort**: Mantener condiciones térmicas aceptables

## Contribuciones Esperadas de la Tesis

### Técnicas:
- ✅ Implementación escalable de MADDPG para control energético
- ✅ Arquitectura descentralizada para ejecución en tiempo real
- ✅ Metodología de entrenamiento eficiente

### Prácticas:
- ✅ Sistema deployable en producción (Docker/Kubernetes)
- ✅ API REST para integración con sistemas BMS/EMS
- ✅ Monitoreo y observabilidad para operación continua

### Científicas:
- 📊 Análisis comparativo con métodos baseline (RBC, MPC)
- 📈 Estudio de escalabilidad con número de edificios
- 🔬 Evaluación de transferibilidad entre comunidades

## Arquitectura del Sistema Implementado

```
┌─────────────────────────────────────────────────────────────┐
│              COMUNIDAD ENERGÉTICA INTERACTIVA                │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Edificio 1│  │Edificio 2│  │Edificio 3│  │Edificio N│   │
│  │          │  │          │  │          │  │          │   │
│  │ Agente 1 │  │ Agente 2 │  │ Agente 3 │  │ Agente N │   │
│  │  (DDPG)  │  │  (DDPG)  │  │  (DDPG)  │  │  (DDPG)  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │             │          │
│       │   Solar     │   HVAC      │  Battery    │          │
│       │   + HVAC    │   + DHW     │  + Loads    │  ...     │
│       └─────────────┴─────────────┴─────────────┘          │
│                          │                                   │
└──────────────────────────┼───────────────────────────────────┘
                           │ Punto de Acople Común
                           ▼
                  ┌─────────────────┐
                  │   RED PÚBLICA   │
                  │   ELÉCTRICA     │
                  │                 │
                  │  - Precios      │
                  │  - Demanda      │
                  │  - Frecuencia   │
                  └─────────────────┘

        Entrenamiento Centralizado (MADDPG)
        ──────────────────────────────────
        ┌──────────────────────────────┐
        │  Crítico Centralizado        │
        │  (observa todos los agentes) │
        └──────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │   Replay Buffer         │
        │   (experiencias multi-  │
        │    agente compartidas)  │
        └─────────────────────────┘
```

## Referencias Teóricas

### Algoritmo MADDPG
- **Paper Original**: "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (Lowe et al., 2017)
- **Extensión**: Aplicación a control continuo multi-agente

### CityLearn
- **Framework**: CityLearn v2.x - Multi-Agent RL for Building Energy Management
- **Challenge**: CityLearn Challenge 2023 Phase 2

### Flexibilidad Energética
- **IEA**: Demand Response and Flexibility Services
- **IEEE**: Smart Grid Communications and Control

## Próximos Pasos de la Investigación

1. **Experimentación**:
   - Entrenar con diferentes configuraciones de comunidad
   - Evaluar con distintas señales de precio/demanda
   - Comparar con baselines (Rule-Based, MPC, Single-Agent)

2. **Validación**:
   - Pruebas con datos reales de edificios
   - Análisis de robustez ante incertidumbre
   - Estudio de transferibilidad

3. **Despliegue**:
   - Integración con sistemas BMS reales
   - Evaluación en testbed o piloto
   - Análisis de impacto económico/ambiental

---

**Esta implementación sirve como base computacional para la investigación doctoral/maestría en control inteligente de comunidades energéticas.**
