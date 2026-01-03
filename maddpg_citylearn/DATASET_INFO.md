# Dataset CityLearn: citylearn_challenge_2022_phase_all_plus_evs

## Descripción del Dataset

Este dataset es parte del **CityLearn Challenge 2022** e incluye características avanzadas específicas para el control de flexibilidad energética en comunidades interactivas, incluyendo **vehículos eléctricos (EVs)**.

## Características del Dataset

### 🏢 Edificios
- **Múltiples edificios** con perfiles heterogéneos (residencial, comercial)
- Cada edificio es un agente autónomo en el sistema multi-agente
- Diferentes patrones de consumo y características térmicas

### 🔋 Recursos de Flexibilidad Energética

#### 1. **Vehículos Eléctricos (EVs)**
- **Cargas controlables diferibles**: Carga puede ser modulada en el tiempo
- **Patrones de uso**: Llegadas/salidas, niveles de carga inicial/final
- **Bidireccionalidad (V2G)**: Potencial de vehicle-to-grid si está habilitado
- **Flexibilidad temporal**: Ventanas de carga entre llegada y salida

#### 2. **Generación Solar Fotovoltaica**
- Generación distribuida en edificios
- Perfiles de irradiancia solar realistas
- Recurso renovable para autoconsumo local

#### 3. **Sistemas de Almacenamiento (Baterías)**
- Baterías eléctricas estacionarias
- Capacidad de carga/descarga controlable
- Estado de carga (SoC) gestionable
- Arbitraje energético y peak shaving

#### 4. **Cargas Térmicas**
- **HVAC** (Heating, Ventilation, Air Conditioning): Control de temperatura con inercia térmica
- **DHW** (Domestic Hot Water): Calentamiento de agua doméstica
- Flexibilidad mediante gestión de setpoints

### 📊 Señales del Entorno

- **Precios de Electricidad**: Tarifas variables para optimización económica
- **Emisiones de Carbono**: Intensidad de carbono de la red eléctrica
- **Condiciones Climáticas**: Temperatura exterior, irradiancia solar
- **Perfiles de Ocupación**: Patrones de uso de edificios

### 🎯 Relevancia para la Tesis

Este dataset es ideal para tu investigación porque incluye:

1. **Múltiples Agentes**: Edificios interactuando como comunidad
2. **Flexibilidad Diversificada**: EVs + baterías + HVAC + solar
3. **Interacción con Red**: Demanda agregada, respuesta a precios
4. **Recursos Distribuidos (DER)**: Generación y almacenamiento local
5. **Cargas Controlables**: EVs como recurso de respuesta a la demanda

## Estructura de Observaciones - CityLearn v2 (42 dimensiones)

Cada agente (edificio) recibe **42 observaciones** en cada timestep:

### Índices de Observaciones

| Índice | Nombre | Descripción | Rango |
|--------|--------|-------------|-------|
| **0** | `solar_generation` | Generación solar PV (kW) | [0, ∞) |
| **1** | `hour` | Hora del día normalizada | [0, 1] |
| **2** | `day_type` | Tipo de día (0=weekday, 1=weekend) | {0, 1} |
| **3** | `daylight_savings_status` | Estado horario de verano | {0, 1} |
| **4** | `month` | Mes normalizado | [0, 1] |
| **5** | `outdoor_dry_bulb_temperature` | Temperatura exterior (°C) | [-20, 50] |
| **6** | `outdoor_relative_humidity` | Humedad relativa exterior (%) | [0, 100] |
| **7** | `electric_vehicle_arrival` | EV llegó este timestep | {0, 1} |
| **8** | `electric_vehicle_availability` | EV disponible/conectado | {0, 1} |
| **9** | `electric_vehicle_charge_rate` | Tasa de carga EV (kW) | [0, 11] |
| **10** | `electric_vehicle_energy_charged` | Energía cargada EV (kWh) | [0, ∞) |
| **11** | `electric_vehicle_state_of_charge` | SoC del EV (%) | [0, 1] |
| **12** | `indoor_dry_bulb_temperature` | Temperatura interior (°C) | [18, 28] |
| **13** | `indoor_dry_bulb_temperature_set_point` | Setpoint temperatura (°C) | [18, 28] |
| **14** | `indoor_relative_humidity` | Humedad relativa interior (%) | [30, 70] |
| **15** | `non_shiftable_load` | Carga eléctrica no controlable (kW) | [0, ∞) |
| **16** | `occupant_count` | Número de ocupantes | [0, N] |
| **17** | `power_outage` | Corte de energía activo | {0, 1} |
| **18** | `cooling_demand` | Demanda de enfriamiento (kWh) | [0, ∞) |
| **19** | `heating_demand` | Demanda de calefacción (kWh) | [0, ∞) |
| **20** | `dhw_demand` | Demanda agua caliente (kWh) | [0, ∞) |
| **21** | `electrical_storage_soc` | SoC batería eléctrica (%) | [0, 1] |
| **22** | `electrical_storage_energy_charged` | Energía cargada batería (kWh) | [0, ∞) |
| **23** | `electrical_storage_energy_discharged` | Energía descargada batería (kWh) | [0, ∞) |
| **24** | `net_electricity_consumption` | Consumo neto (kW) | (-∞, ∞) |
| **25** | `carbon_intensity` | Intensidad carbono red (kg CO₂/kWh) | [0, 1] |
| **26** | `electricity_pricing` | Precio electricidad ($/kWh) | [0, 0.5] |
| **27** | `electricity_pricing_predicted_1h` | Precio predicho +1h | [0, 0.5] |
| **28** | `electricity_pricing_predicted_2h` | Precio predicho +2h | [0, 0.5] |
| **29** | `electricity_pricing_predicted_3h` | Precio predicho +3h | [0, 0.5] |
| **30** | `electricity_pricing_predicted_6h` | Precio predicho +6h | [0, 0.5] |
| **31** | `electricity_pricing_predicted_12h` | Precio predicho +12h | [0, 0.5] |
| **32** | `electricity_pricing_predicted_24h` | Precio predicho +24h | [0, 0.5] |
| **33** | `cooling_storage_soc` | SoC almacenamiento frío (%) | [0, 1] |
| **34** | `dhw_storage_soc` | SoC almacenamiento DHW (%) | [0, 1] |
| **35** | `indoor_dry_bulb_temperature_delta` | Δ temperatura vs setpoint (°C) | [-5, 5] |
| **36** | `indoor_dry_bulb_temperature_delta_rolling_12h` | Δ temperatura 12h (°C) | [-3, 3] |
| **37** | `indoor_dry_bulb_temperature_delta_rolling_24h` | Δ temperatura 24h (°C) | [-3, 3] |
| **38** | `net_electricity_consumption_rolling_12h` | Consumo medio 12h (kW) | [0, ∞) |
| **39** | `net_electricity_consumption_rolling_24h` | Consumo medio 24h (kW) | [0, ∞) |
| **40** | `net_electricity_consumption_predicted_1h` | Consumo predicho +1h (kW) | [0, ∞) |
| **41** | `net_electricity_consumption_predicted_24h` | Consumo predicho +24h (kW) | [0, ∞) |

### Categorías de Observaciones

**🌞 Generación y Clima** (obs 0-6):
- Solar PV, hora, día, mes, temperatura, humedad

**🚗 Vehículo Eléctrico** (obs 7-11):
- Llegada, disponibilidad, tasa de carga, energía cargada, SoC

**🏠 Condiciones Interiores** (obs 12-17):
- Temperatura, setpoint, humedad, ocupantes, cortes

**🔥 Demandas Térmicas** (obs 18-20):
- Enfriamiento, calefacción, agua caliente

**🔋 Almacenamiento** (obs 21-23, 33-34):
- Batería eléctrica (SoC, carga, descarga)
- Almacenamiento térmico (frío, DHW)

**⚡ Consumo y Red** (obs 24-32):
- Consumo neto, carbono, precio actual y predicciones

**📊 Historial y Predicciones** (obs 35-41):
- Δ temperatura (instantáneo, 12h, 24h)
- Consumo medio y predicciones

## Estructura de Acciones - CityLearn v2 (3 dimensiones)

Cada agente controla **3 acciones continuas**:

### Acciones Disponibles

| Índice | Nombre | Descripción | Rango | Efecto |
|--------|--------|-------------|-------|--------|
| **0** | `electrical_storage` | Control batería eléctrica | [-1, 1] | -1: descarga máxima, 0: sin acción, +1: carga máxima |
| **1** | `cooling_device` | Control enfriamiento (HVAC) | [-1, 1] | Ajuste del setpoint de temperatura (más bajo = más enfriamiento) |
| **2** | `dhw_storage` | Control agua caliente | [-1, 1] | Ajuste del setpoint de DHW (más alto = más calentamiento) |

### Detalles de Acciones

#### Acción 0: `electrical_storage` (Batería)
- **-1.0**: Descarga al máximo C-rate (típicamente 0.25C = 25% capacidad/hora)
- **0.0**: Sin carga ni descarga (idle)
- **+1.0**: Carga al máximo C-rate
- **Capacidad típica**: 6.4 kWh por batería
- **C-rate MADDPG**: 0.25 (1.6 kW máximo)
- **C-rate MARLISA**: 0.15 (0.96 kW máximo)

#### Acción 1: `cooling_device` (HVAC)
- **-1.0**: Enfriar agresivamente (reducir setpoint al mínimo)
- **0.0**: Mantener setpoint actual
- **+1.0**: Reducir enfriamiento (aumentar setpoint al máximo)
- **Rango setpoint**: típicamente 20-26°C
- **Reducción MADDPG**: hasta 30% de la demanda de enfriamiento
- **Reducción MARLISA**: hasta 15% de la demanda

#### Acción 2: `dhw_storage` (Agua Caliente)
- **-1.0**: Reducir temperatura DHW al mínimo
- **0.0**: Mantener setpoint actual
- **+1.0**: Aumentar temperatura DHW al máximo
- **Rango setpoint**: típicamente 50-70°C
- **Inercia térmica**: el agua caliente mantiene temperatura varias horas
- **Reducción MADDPG**: hasta 20% de la demanda DHW
- **Reducción MARLISA**: sin control (0%)

### Nota sobre EV Charging

**⚠️ Importante**: Aunque hay observaciones de EV (obs 7-11), **no hay acción explícita de control de EV** en este schema. El control de EV se realiza indirectamente:

- La carga del EV se considera parte del `non_shiftable_load` (obs 15)
- El control se logra mediante:
  - **Batería**: Usar batería estacionaria cuando EV está cargando
  - **Solar**: Maximizar autoconsumo solar durante ventanas de carga
  - **Precio**: Coordinar con señales de precio para carga óptima

Para **V2G (Vehicle-to-Grid)**, se puede simular usando:
```python
# Pseudo-código para V2G simulado
if ev_available and ev_soc > 0.3 and electricity_price > threshold:
    # Usar batería para simular descarga V2G
    battery_action = -1.0  # Descarga batería
```

**Nota**: Las acciones están normalizadas en rango [-1, 1] y CityLearn las escala internamente a los límites físicos de cada actuador.

## Métricas de Evaluación

CityLearn evalúa el desempeño usando múltiples KPIs:

### 1. **Económicos**
- **Costo Total**: Suma de costos de electricidad
- **Ahorro vs Baseline**: Comparación con estrategia sin control

### 2. **Demanda de Red**
- **Peak Demand**: Máxima demanda agregada
- **Peak-to-Average Ratio**: Factor de carga
- **Ramping**: Cambios abruptos en demanda

### 3. **Ambientales**
- **Emisiones de CO₂**: Toneladas de carbono
- **Uso de Renovables**: Autoconsumo solar

### 4. **Confort**
- **Violaciones de Temperatura**: Desviaciones del rango confortable
- **Disconfort Térmico**: Penalizaciones por temperatura inadecuada

## Uso en el Código

El dataset se importa automáticamente desde CityLearn:

```python
from maddpg_tesis.envs.citylearn_env import CityLearnMultiAgentEnv

# El schema se carga desde los datasets incluidos en CityLearn
env = CityLearnMultiAgentEnv(
    schema="citylearn_challenge_2022_phase_all_plus_evs",
    central_agent=False
)

print(f"Número de agentes (edificios): {env.n_agents}")
print(f"Dimensión de observación: {env.obs_dim}")
print(f"Dimensión de acción: {env.action_dim}")
```

## 17 Edificios del Dataset - Recursos Reales Heterogéneos

El dataset `citylearn_challenge_2022_phase_all_plus_evs` incluye **17 edificios comerciales** con **recursos heterogéneos**:

### Tabla de Edificios y Recursos Específicos

| ID | Nombre | Solar PV (kW) | Batería (kWh) | EV Charger | HVAC | DHW | Obs Dim | Action Dim | Perfil |
|----|--------|---------------|---------------|------------|------|-----|---------|------------|---------|
| 0 | Building_1 | 12.0 | 6.4 | ✅ (1) | ❌ | ❌ | 37 | 3 | **Batería + EV charger + washing** |
| 1 | Building_2 | 4.0 | 6.4 | ❌ | ❌ | ❌ | 28 | 1 | **Solo batería** |
| 2 | Building_3 | 4.0 | 6.4 | ❌ | ❌ | ❌ | 28 | 1 | **Solo batería** |
| 3 | Building_4 | 8.0 | 6.4 | ✅ (1) | ❌ | ❌ | 35 | 2 | **Batería + EV charger** |
| 4 | Building_5 | 10.0 | 6.4 | ✅ (1) | ❌ | ❌ | 35 | 2 | **Batería + EV charger** |
| 5 | Building_6 | 4.0 | 6.4 | ❌ | ❌ | ❌ | 28 | 1 | **Solo batería** |
| 6 | Building_7 | 9.0 | 6.4 | ✅ (1) | ❌ | ❌ | 35 | 2 | **Batería + EV charger** |
| 7 | Building_8 | 4.0 | 6.4 | ❌ | ❌ | ❌ | 28 | 1 | **Solo batería** |
| 8 | Building_9 | 4.0 | 6.4 | ❌ | ❌ | ❌ | 28 | 1 | **Solo batería** |
| 9 | Building_10 | 6.0 | 6.4 | ✅ (1) | ❌ | ❌ | 35 | 2 | **Batería + EV charger** |
| 10 | Building_11 | 5.0 | 6.4 | ❌ | ❌ | ❌ | 28 | 1 | **Solo batería** |
| 11 | Building_12 | 8.0 | 6.4 | ✅ (1) | ❌ | ❌ | 35 | 2 | **Batería + EV charger** |
| 12 | Building_13 | 5.0 | 6.4 | ❌ | ❌ | ❌ | 28 | 1 | **Solo batería** |
| 13 | Building_14 | 5.0 | 6.4 | ❌ | ❌ | ❌ | 28 | 1 | **Solo batería** |
| 14 | Building_15 | 15.0 | 6.4 | ✅ (2) | ❌ | ❌ | 42 | 3 | **Batería + EV chargers×2** |
| 15 | Building_16 | 5.0 | 6.4 | ❌ | ❌ | ❌ | 28 | 1 | **Solo batería** |
| 16 | Building_17 | 5.0 | 6.4 | ❌ | ❌ | ❌ | 28 | 1 | **Solo batería** |

### Análisis de Recursos Disponibles

#### ☀️ **Solar PV (Todos los edificios tienen, pero diferente capacidad)**
- **Rango**: 4-15 kW nominal
- **Edificios pequeños** (4-5 kW): 2, 3, 6, 8, 9, 11, 13, 14, 16, 17 → 10 edificios
- **Edificios medianos** (6-10 kW): 4, 5, 7, 10, 12 → 5 edificios  
- **Edificios grandes** (12-15 kW): 1, 15 → 2 edificios
- **Promedio**: 6.59 kW
- **Total instalado**: 112 kW (comunidad)

#### 🔋 **Batería Eléctrica (Todos los edificios tienen - UNIFORME)**
- **Capacidad**: 6.4 kWh (idéntico en todos)
- **Potencia máxima**: 5.0 kW (idéntico en todos)
- **Eficiencia**: 90% (idéntico en todos)
- **Total comunidad**: 108.8 kWh

#### 🚗 **Cargadores de Vehículos Eléctricos (7 edificios tienen)**
- **Edificios CON cargadores EV**: 1, 4, 5, 7, 10, 12, 15 → **7 de 17** (41%)
- **Building_15**: 2 cargadores EV (único con doble cargador)
- **Resto**: 1 cargador por edificio
- ⚠️ **IMPORTANTE**: Los cargadores están como **ACCIONES** (control de carga)
- ❌ **Observaciones EV NO disponibles**: No hay SoC, availability, charge_rate observables
- 🤔 **Control "a ciegas"**: Se controla la carga sin retroalimentación del estado del EV

#### ❄️ **Cooling Storage (Deshabilitado)**
- **Objetos cooling_storage**: ✅ 17/17 edificios tienen el objeto
- **Capacidad**: ❌ 0.0 kWh en TODOS → **DESHABILITADO**
- **Estado**: Storage existe pero sin capacidad útil
- **Implicación**: No se puede almacenar frío, solo control directo HVAC si existiera

#### 🔥 **Heating Storage (Deshabilitado)**
- **Objetos heating_storage**: ✅ 17/17 edificios tienen el objeto
- **Capacidad**: ❌ 0.0 kWh en TODOS → **DESHABILITADO**
- **Estado**: Storage existe pero sin capacidad útil
- **Implicación**: No se puede almacenar calor para calefacción

#### 🚿 **DHW Storage (Deshabilitado)**
- **Objetos dhw_storage**: ✅ 17/17 edificios tienen el objeto
- **Capacidad**: ❌ 0.0 kWh en TODOS → **DESHABILITADO**
- **Estado**: Storage existe pero sin capacidad útil
- **Implicación**: No se puede almacenar agua caliente

#### 🧺 **Washing Machine (Solo Building_1)**
- **Edificios CON washing**: 1 → **1 de 17** (6%)
- **Tipo**: Carga diferible (puede desplazarse en el tiempo)
- **Flexibilidad**: Demand Response básico
- **Uso limitado**: Solo para análisis de Building_1

### Perfiles de Acción por Edificio

#### **Perfil 1: Solo Batería** (1 acción) - **10 edificios**
- Edificios: 2, 3, 6, 8, 9, 11, 13, 14, 16, 17
- Acciones: `[electrical_storage]`
- Obs dim: 28
- Control: Solo arbitraje de batería eléctrica

#### **Perfil 2: Batería + EV Charger** (2 acciones) - **5 edificios**
- Edificios: 4, 5, 7, 10, 12
- Acciones: `[electrical_storage, ev_charger]`
- Obs dim: 35
- Control: Batería + carga de vehículo eléctrico

#### **Perfil 3: Batería + EV Charger + Otros** (3 acciones) - **2 edificios**
- **Building_1**: `[electrical_storage, ev_charger, washing_machine]`
  - Obs dim: 37
  - Control: Batería + EV + lavadora
- **Building_15**: `[electrical_storage, ev_charger_1, ev_charger_2]`
  - Obs dim: 42
  - Control: Batería + 2 cargadores EV (único con doble cargador)

### Recursos Disponibles (Resumen Real)

### Recursos Disponibles (Resumen Real)

1. **☀️ Solar PV**: **17/17 edificios** (100%) - HETEROGÉNEO
   - Capacidad variable: 4-15 kW
   - Generación depende de irradiancia solar
   - Autoconsumo prioritario

2. **🔋 Batería Eléctrica**: **17/17 edificios** (100%) - HOMOGÉNEO
   - Capacidad: 6.4 kWh (uniforme)
   - Potencia: 5.0 kW (uniforme)
   - Eficiencia: 90% (uniforme)
   - **Único recurso presente en TODOS los edificios con misma especificación**

3. **🚗 Cargadores de Vehículos Eléctricos**: **7/17 edificios** (41%) - HETEROGÉNEO ✅ ACTIVO
   - Edificios: 1, 4, 5, 7, 10, 12, 15
   - Control de carga EV (acción continua)
   - Building_15: 2 cargadores (único)
   - ⚠️ Sin observaciones de estado del EV (control sin retroalimentación)
   - **Dentro del alcance de flexibilidad**: Demand Response EV ✅

4. **❄️ Cooling Storage**: **17/17 edificios** (100%) - DESHABILITADO ❌
   - Todos los edificios tienen objeto cooling_storage
   - Capacidad: 0.0 kWh → SIN almacenamiento térmico
   - No útil para flexibilidad en este schema

5. **🔥 Heating Storage**: **17/17 edificios** (100%) - DESHABILITADO ❌
   - Todos los edificios tienen objeto heating_storage
   - Capacidad: 0.0 kWh → SIN almacenamiento térmico
   - No útil para flexibilidad en este schema

6. **🚿 DHW Storage**: **17/17 edificios** (100%) - DESHABILITADO ❌
   - Todos los edificios tienen objeto dhw_storage
   - Capacidad: 0.0 kWh → SIN almacenamiento agua caliente
   - No útil para flexibilidad en este schema

7. **🧺 Washing Machines**: **1/17 edificios** (6%) - MUY LIMITADO ✅ ACTIVO
   - Solo Building_1 tiene control de lavadora
   - Carga diferible (washing_machine_1)
   - **Dentro del alcance**: Demand Response básico ✅

### Dimensiones por Edificio (Correctas)

### Dimensiones por Edificio (Correctas)

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| **Observaciones** | 28, 35, 37, 42 | **VARIABLE por edificio** según recursos |
| **Acciones** | 1, 2, 3 | **VARIABLE: 1 (solo bat), 2 (bat+cool), 3 (bat+cool+dhw)** |
| **Espacio observación** | Continuo | Normalizado, dimensión variable |
| **Espacio acción** | [-1, 1]ⁿ | Continuo, n = 1, 2, o 3 según edificio |

### Resumen Multi-Agente (Padded para MADDPG)

```python
# Dimensiones globales MADDPG con padding
n_agents = 17                     # Un agente por edificio
obs_dim_max = 42                  # Máxima dimensión (Building_15)
action_dim_max = 3                # Máxima dimensión (Buildings 1, 15)

# Padding requerido
# Edificios con menos dimensiones se rellenan con 0s
# Ejemplo: Building_2 (28 obs, 1 action) → (42 obs padded, 3 actions padded)

# Estado conjunto (usado por crítico centralizado)
global_obs_dim = 17 × 42 = 714    # Todas las observaciones (padded)
global_action_dim = 17 × 3 = 51   # Todas las acciones (padded)

# Shapes con padding
observations.shape = (17, 42)     # Matriz padded
actions.shape = (17, 3)           # Matriz padded
```

### Distribución Real de Dimensiones

**Observaciones**:
- 28 dim: 10 edificios (59%)
- 35 dim: 5 edificios (29%)
- 37 dim: 1 edificio (6%)
- 42 dim: 1 edificio (6%)

**Acciones**:
- 1 acción: 10 edificios (59%) → Solo batería
- 2 acciones: 5 edificios (29%) → Batería + cooling
- 3 acciones: 2 edificios (12%) → Batería + cooling + dhw

### Heterogeneidad Real de Edificios

Los edificios difieren significativamente en:

1. **📊 Capacidad Solar PV**: 
   - Rango: 4-15 kW (factor 3.75x entre mínimo y máximo)
   - Building_15 (15 kW): 3.75× más que Building_2 (4 kW)
   - Impacta capacidad de autoconsumo y exportación

2. **🎮 Recursos Controlables**:
   - **10 edificios** (59%): Solo batería
   - **5 edificios** (29%): Batería + cooling
   - **2 edificios** (12%): Batería + cooling + DHW
   - Asimetría en capacidad de control

3. **📐 Dimensionalidad**:
   - Observaciones: 28-42 (diferencia de 14 dimensiones, 50%)
   - Acciones: 1-3 (algunos edificios tienen 3× más control)
   - Requiere padding para MADDPG

4. **🏠 Perfiles de Carga**: 
   - Buildings 1, 15: Mayor complejidad (cooling + DHW)
   - Buildings 2, 3, 6, 8, 9, 11, 13, 14, 16, 17: Solo gestión eléctrica
   - Buildings 4, 5, 7, 10, 12: Gestión eléctrica + térmica

5. **⚖️ Importancia en Coordinación**:
   - Buildings 1, 15: "Hub energéticos" (más recursos, más opciones)
   - Mayoría: Agentes simplificados (solo batería)
   - Coordinación asimétrica: edificios complejos lideran, simples siguen

Esta **heterogeneidad real** hace que:
- La coordinación multi-agente sea **asimétrica** (no todos contribuyen igual)
- El padding sea necesario para arquitectura MADDPG uniforme
- Los edificios con más recursos (1, 15) tengan mayor impacto en optimización agregada
- La mayoría (59%) solo pueda hacer arbitraje de batería, limitando flexibilidad térmica

---

## 🎯 Alcance de la Tesis: Control de Flexibilidad Energética con MADRL

### Recursos ACTIVOS y Utilizables para Flexibilidad

#### ✅ **CORE - Recursos Principales** (100% dentro del alcance)

1. **☀️ Solar PV - Generación Distribuida**: 17/17 edificios (100%)
   - Capacidad: 4-15 kW (heterogéneo)
   - **Flexibilidad**: Generación renovable, autoconsumo local
   - **Control**: No directamente controlable, pero observable para coordinación
   - **Impacto**: Optimización de autoconsumo vs exportación
   - ✅ **DENTRO DEL ALCANCE**: Gestión de generación distribuida

2. **🔋 Batería Eléctrica - Almacenamiento**: 17/17 edificios (100%)
   - Capacidad: 6.4 kWh (uniforme)
   - **Flexibilidad**: Arbitraje energético, peak shaving, autoconsumo solar
   - **Control**: Carga/descarga continua [-1, 1]
   - **Impacto**: Recurso CORE, presente en TODOS los agentes
   - ✅ **DENTRO DEL ALCANCE**: Gestión de almacenamiento eléctrico

3. **🚗 EV Chargers - Demand Response**: 7/17 edificios (41%)
   - Cargadores: 1-2 por edificio (Building_15 tiene 2)
   - **Flexibilidad**: Cargas flexibles, desplazamiento temporal
   - **Control**: Modulación de tasa de carga continua
   - **Impacto**: Demand Response sin retroalimentación (control "a ciegas")
   - ✅ **DENTRO DEL ALCANCE**: Gestión de cargas flexibles EV

#### 🎁 **BONUS - Recursos Secundarios**

4. **🧺 Washing Machine - Carga Diferible**: 1/17 edificios (6%)
   - Solo Building_1
   - **Flexibilidad**: Desplazamiento temporal de carga
   - **Control**: Modulación de operación
   - **Impacto**: Limitado, solo 1 edificio
   - ✅ **DENTRO DEL ALCANCE**: Demand Response básico (bonus)

#### ❌ **FUERA DEL ALCANCE - Recursos Deshabilitados**

5. **❄️ Cooling Storage**: 0.0 kWh → **NO UTILIZABLE**
   - Objetos existen pero capacidad = 0
   - No aporta flexibilidad térmica
   - ❌ **FUERA DEL ALCANCE**: Sin almacenamiento térmico frío

6. **🔥 Heating Storage**: 0.0 kWh → **NO UTILIZABLE**
   - Objetos existen pero capacidad = 0
   - No aporta flexibilidad térmica
   - ❌ **FUERA DEL ALCANCE**: Sin almacenamiento térmico calor

7. **🚿 DHW Storage**: 0.0 kWh → **NO UTILIZABLE**
   - Objetos existen pero capacidad = 0
   - No aporta flexibilidad de agua caliente
   - ❌ **FUERA DEL ALCANCE**: Sin almacenamiento DHW

### Resumen de Validez para la Tesis

#### 📊 Métricas de Flexibilidad Disponible

```
Recursos CORE activos:        3/3 (100%) ✅
├─ Solar PV:                  17/17 edificios ✅
├─ Batería eléctrica:         17/17 edificios ✅
└─ EV Chargers:               7/17 edificios ✅

Recursos BONUS activos:       1/4 (25%) 🎁
└─ Washing machines:          1/17 edificios ✅

Recursos deshabilitados:      3/7 (43%) ❌
├─ Cooling storage:           0 kWh (sin capacidad)
├─ Heating storage:           0 kWh (sin capacidad)
└─ DHW storage:               0 kWh (sin capacidad)
```

#### ✅ **CONCLUSIÓN: Dataset VÁLIDO para Tesis MADRL**

**Justificación**:

1. **Recursos CORE al 100%**: Todos los recursos principales de flexibilidad energética están activos
   - Generación distribuida (Solar PV)
   - Almacenamiento eléctrico (Baterías)
   - Demand Response (EV Chargers)

2. **Multi-Agente Real**: 17 agentes con heterogeneidad real
   - Diferentes capacidades solares (4-15 kW)
   - Diferentes recursos de control (1-3 acciones)
   - Coordinación asimétrica necesaria

3. **Flexibilidad Diversificada**:
   - Generación: 100% edificios con solar
   - Almacenamiento: 100% edificios con batería
   - Cargas flexibles: 41% edificios con EV chargers
   - Bonus: 6% con washing machines

4. **Desafíos MADRL Presentes**:
   - Coordinación multi-agente (17 agentes)
   - Espacios de acción continuos
   - Observaciones parciales (POMDP)
   - Control sin retroalimentación (EV chargers)
   - Heterogeneidad de recursos

5. **Objetivos de Optimización Claros**:
   - Minimizar costos energéticos
   - Reducir picos de demanda (peak shaving)
   - Maximizar autoconsumo solar
   - Reducir emisiones de carbono
   - Gestionar flexibilidad EV

**Limitaciones Reconocidas**:
- ❌ Sin flexibilidad térmica activa (cooling/heating/dhw storage = 0)
- ⚠️ Control EV sin observaciones de estado ("a ciegas")
- ⚠️ Washing machine solo en 1 edificio (impacto limitado)

**Pero estas limitaciones NO invalidan el dataset porque**:
- Los recursos CORE de flexibilidad eléctrica están al 100%
- La flexibilidad eléctrica es el CORE de la gestión energética moderna
- EV Demand Response es altamente relevante (7/17 edificios)
- La ausencia de flexibilidad térmica simplifica el problema sin quitarle validez

### 🎓 Validación para Publicación

Este dataset es **suficiente y apropiado** para:

✅ **Tesis de maestría/doctorado** en control de flexibilidad energética  
✅ **Publicaciones científicas** sobre MADRL en sistemas energéticos  
✅ **Comparación con baselines** (MARLISA, RBC, etc.)  
✅ **Demostración de coordinación multi-agente** en comunidades energéticas  
✅ **Gestión de recursos distribuidos** (DER - Distributed Energy Resources)  
✅ **Demand Response con EVs** (tema de alta relevancia actual)  

**Referencias que validan este tipo de dataset**:
- CityLearn Challenge 2022 (competencia internacional)
- Papers sobre MARLISA, MADDPG en CityLearn
- Investigación sobre gestión de comunidades energéticas
- Estudios sobre integración de EVs en redes inteligentes

## Consideraciones Técnicas

### Dimensionalidad Completa
- **Observaciones por agente**: 28-42 características continuas (**VARIABLE**)
- **Acciones por agente**: 1-3 acciones continuas en [-1, 1] (**VARIABLE**)
- **Número de agentes**: 17 edificios
- **Estado conjunto (padded)**: 714 dimensiones (17 × 42) - usado por crítico centralizado MADDPG
- **Acción conjunta (padded)**: 51 dimensiones (17 × 3) - espacio de acción global
- **Horizonte temporal**: 8760 timesteps (1 año completo, datos horarios)
- **Padding requerido**: Sí, para uniformizar dimensiones entre agentes heterogéneos

### Desafíos de Aprendizaje
1. **Curse of Dimensionality**: Estado conjunto crece con número de agentes
2. **Exploración**: Balance entre exploración y explotación en acciones continuas
3. **Credit Assignment**: Atribuir recompensas a acciones individuales en contexto multi-agente
4. **No Estacionariedad**: Políticas de otros agentes cambian durante entrenamiento

### Ventajas del Dataset para MADDPG
- ✅ **Acciones Continuas**: Ideal para DDPG (base de MADDPG)
- ✅ **Multi-Agente**: Múltiples edificios coordinando
- ✅ **Cooperativo**: Objetivo común (minimizar costo/demanda agregada)
- ✅ **Realista**: Datos basados en edificios y clima reales

## Validación del Dataset

Para verificar que el dataset está correctamente instalado:

```python
from citylearn.citylearn import CityLearnEnv

# Listar schemas disponibles
from citylearn.data import DataSet

# Verificar que el schema existe
try:
    env = CityLearnEnv(schema="citylearn_challenge_2022_phase_all_plus_evs")
    print("✓ Dataset cargado exitosamente")
    print(f"  - Edificios: {len(env.buildings)}")
    print(f"  - Timesteps: {env.time_steps}")
except Exception as e:
    print(f"✗ Error: {e}")
```

## Referencias

- **CityLearn Documentation**: https://intelligent-environments-lab.github.io/CityLearn/
- **Challenge 2022**: Información sobre el reto y dataset
- **Paper**: "CityLearn v2: Energy-Flexible, Grid-Interactive Demand Response"

---

**Este dataset es la base de datos principal para tu tesis sobre control de flexibilidad energética con MADDPG, incorporando vehículos eléctricos como elementos clave de flexibilidad.**
