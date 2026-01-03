"""
Funciones de recompensa personalizadas para CityLearn v2.
Optimizan las 5 métricas principales: Cost, Carbon Emissions, Ramping, Load Factor, Electricity Consumption.

Basado en la documentación oficial de CityLearn:
https://www.citylearn.net/overview/cost_function.html

PARADIGMA CTDE COOPERATIVO:
- Centralized Training: Critic ve estado global de todos los agentes
- Decentralized Execution: Cada actor actúa con observación local
- Team Reward: Todos los agentes reciben la MISMA recompensa global
  para incentivar cooperación en lugar de comportamiento egoísta
"""
from typing import Any, List, Mapping, Union, Optional
import numpy as np


class MADDPG_Reward_Function:
    """
    Función de recompensa COOPERATIVA para MADDPG (CTDE).
    
    IMPORTANTE: En MADRL cooperativo, todos los agentes reciben la MISMA
    recompensa global (team reward). Esto incentiva:
    - Cooperación entre edificios
    - Optimización del bienestar del distrito completo
    - Evita comportamiento egoísta de agentes individuales
    
    Métricas optimizadas:
    1. Cost (Costo de electricidad) - Minimizar gastos del distrito
    2. Carbon Emissions (Emisiones CO₂) - Minimizar huella de carbono total
    3. Ramping (Estabilidad de red) - Minimizar cambios bruscos agregados
    4. Load Factor (Factor de carga) - Maximizar uniformidad del distrito
    5. Electricity Consumption - Minimizar consumo total del distrito
    
    Arquitectura CTDE:
    - Entrenamiento: Critic centralizado ve obs/actions de TODOS los edificios
    - Ejecución: Cada actor usa SOLO su observación local
    - Recompensa: TEAM REWARD compartida = cooperación
    """
    
    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        weights: Optional[Mapping[str, float]] = None,
        cooperative: bool = True  # CTDE cooperativo por defecto
    ):
        """
        Inicializa la función de recompensa cooperativa.
        
        Args:
            env_metadata: Metadatos del entorno CityLearn
            weights: Pesos para cada métrica
            cooperative: Si True, usa team reward (misma para todos)
                        Si False, usa recompensas individuales
        """
        self.env_metadata = env_metadata
        self.central_agent = env_metadata.get('central_agent', False)
        self.cooperative = cooperative  # CTDE cooperativo
        
        # Pesos por defecto balanceados para las 5 métricas principales
        default_weights = {
            'cost': 0.20,                    # 20% - Costo de electricidad
            'carbon': 0.20,                  # 20% - Emisiones de carbono
            'ramping': 0.20,                 # 20% - Estabilidad de red
            'load_factor': 0.20,             # 20% - Factor de carga
            'electricity_consumption': 0.20, # 20% - Consumo eléctrico total
        }
        self.weights = weights if weights is not None else default_weights
        
        # Normalizar pesos
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # Variables para tracking temporal (necesario para ramping)
        self._prev_net_consumption: Optional[List[float]] = None
        self._prev_total_consumption: Optional[float] = None
        self._episode_consumption: List[List[float]] = []
        
    def reset(self) -> None:
        """Resetea variables al inicio de cada episodio."""
        self._prev_net_consumption = None
        self._prev_total_consumption = None
        self._episode_consumption = []
    
    def calculate(
        self, 
        observations: List[Mapping[str, Union[int, float]]]
    ) -> List[float]:
        """
        Calcula la recompensa COOPERATIVA (team reward) para CTDE.
        
        En MADRL cooperativo, todos los agentes reciben la MISMA recompensa
        basada en el rendimiento GLOBAL del distrito, no individual.
        
        Esto es fundamental para que los agentes aprendan a cooperar.
        
        Args:
            observations: Lista de observaciones de cada edificio
                
        Returns:
            Lista con la MISMA recompensa global para todos los agentes (team reward)
        """
        n_buildings = len(observations)
        
        # ================================================================
        # PASO 1: CALCULAR MÉTRICAS GLOBALES DEL DISTRITO (no individuales)
        # ================================================================
        
        # Extraer datos de todos los edificios
        consumptions = []
        costs = []
        emissions = []
        storage_socs = []
        
        for obs in observations:
            net_consumption = float(obs.get('net_electricity_consumption', 0.0))
            non_shiftable = float(obs.get('non_shiftable_load', 0.0))
            consumption = max(net_consumption, non_shiftable)
            consumptions.append(consumption)
            
            pricing = float(obs.get('electricity_pricing', 0.15))
            carbon_intensity = float(obs.get('carbon_intensity', 0.2))
            
            costs.append(consumption * pricing)
            emissions.append(consumption * carbon_intensity)
            storage_socs.append(float(obs.get('electrical_storage_soc', 0.5)))
        
        # ================================================================
        # PASO 2: CALCULAR TEAM REWARD BASADA EN MÉTRICAS GLOBALES
        # ================================================================
        
        # Métricas AGREGADAS del distrito (lo que importa en CTDE cooperativo)
        total_consumption = sum(consumptions)
        total_cost = sum(costs)
        total_emissions = sum(emissions)
        mean_consumption = np.mean(consumptions) if consumptions else 0.0
        
        # === MÉTRICA 1: COSTO TOTAL DEL DISTRITO ===
        # Normalizado por número de edificios
        cost_penalty = total_cost / max(n_buildings, 1)
        
        # === MÉTRICA 2: EMISIONES TOTALES DEL DISTRITO ===
        carbon_penalty = total_emissions / max(n_buildings, 1)
        
        # === MÉTRICA 3: RAMPING GLOBAL (cambio en consumo total) ===
        if self._prev_total_consumption is not None:
            ramping_penalty = abs(total_consumption - self._prev_total_consumption) / max(n_buildings * 5.0, 1.0)
        else:
            ramping_penalty = 0.0
        
        # === MÉTRICA 4: LOAD FACTOR DEL DISTRITO ===
        # Penalizar varianza en consumo (queremos uniformidad)
        consumption_variance = np.var(consumptions) if len(consumptions) > 1 else 0.0
        load_factor_penalty = np.sqrt(consumption_variance) / max(mean_consumption, 1.0)
        
        # === MÉTRICA 5: CONSUMO TOTAL DE ELECTRICIDAD ===
        consumption_penalty = total_consumption / max(n_buildings * 10.0, 1.0)
        
        # ================================================================
        # PASO 3: COMBINAR EN TEAM REWARD
        # ================================================================
        
        team_penalty = (
            self.weights['cost'] * cost_penalty +
            self.weights['carbon'] * carbon_penalty +
            self.weights['ramping'] * ramping_penalty +
            self.weights['load_factor'] * load_factor_penalty +
            self.weights.get('electricity_consumption', 0.0) * consumption_penalty
        )
        
        # === BONUS POR COORDINACIÓN (SOC uniforme entre edificios) ===
        soc_std = np.std(storage_socs) if len(storage_socs) > 1 else 0.0
        if soc_std < 0.15:  # SOCs muy uniformes = buena coordinación
            team_penalty -= 0.1 * (0.15 - soc_std)
        
        # === PENALTY POR PICOS EXTREMOS DEL DISTRITO ===
        peak_threshold = mean_consumption * 1.5
        max_consumption = max(consumptions) if consumptions else 0.0
        if max_consumption > peak_threshold:
            peak_penalty = (max_consumption - peak_threshold) / peak_threshold
            team_penalty += 0.1 * peak_penalty
        
        # Actualizar historial
        self._prev_net_consumption = consumptions
        self._prev_total_consumption = total_consumption
        self._episode_consumption.append(consumptions)
        
        # ================================================================
        # PASO 4: DEVOLVER TEAM REWARD (MISMA PARA TODOS LOS AGENTES)
        # ================================================================
        # En CTDE cooperativo, TODOS reciben la MISMA recompensa
        team_reward = -team_penalty
        
        if self.cooperative:
            # CTDE cooperativo: misma recompensa para todos
            return [team_reward] * n_buildings
        else:
            # Modo legacy: recompensas individuales (no recomendado)
            return [team_reward] * n_buildings
    
    def _get_cost_penalty(self, obs: Mapping[str, Union[int, float]]) -> float:
        """
        Calcula penalización por costo de electricidad.
        
        Usa el costo neto del consumo eléctrico, considerando:
        - Precio dinámico de electricidad (time-of-use)
        - Carga no desplazable como proxy de consumo base
        """
        # Consumo neto de electricidad (puede ser 0 si hay storage)
        net_consumption = float(obs.get('net_electricity_consumption', 0.0))
        
        # Carga no desplazable (siempre presente)
        non_shiftable = float(obs.get('non_shiftable_load', 0.0))
        
        # Usar el mayor de los dos como indicador de consumo
        consumption = max(net_consumption, non_shiftable)
        
        # Precio de electricidad ($/kWh)
        pricing = float(obs.get('electricity_pricing', 0.15))
        
        # Costo = consumo * precio
        cost = consumption * pricing
        
        return cost
    
    def _get_carbon_penalty(self, obs: Mapping[str, Union[int, float]]) -> float:
        """
        Calcula penalización por emisiones de CO₂.
        
        Usa las emisiones netas del consumo eléctrico, considerando:
        - Intensidad de carbono de la red (varía por hora)
        - Carga no desplazable como proxy de consumo
        """
        # Consumo neto de electricidad
        net_consumption = float(obs.get('net_electricity_consumption', 0.0))
        
        # Carga no desplazable
        non_shiftable = float(obs.get('non_shiftable_load', 0.0))
        
        # Usar el mayor como indicador
        consumption = max(net_consumption, non_shiftable)
        
        # Intensidad de carbono (kg CO₂/kWh)
        carbon_intensity = float(obs.get('carbon_intensity', 0.2))
        
        # Emisiones = consumo * intensidad
        emissions = consumption * carbon_intensity
        
        return emissions
    
    def _get_ramping_penalty(self, building_idx: int, current_consumption: float) -> float:
        """
        Calcula penalización por ramping (cambios bruscos de consumo).
        
        Ramping alto indica inestabilidad en la red eléctrica.
        Se penalizan cambios grandes entre timesteps consecutivos.
        
        Args:
            building_idx: Índice del edificio
            current_consumption: Consumo actual del edificio (non_shiftable_load)
        """
        if self._prev_net_consumption is None:
            return 0.0
        
        if building_idx >= len(self._prev_net_consumption):
            return 0.0
        
        prev_consumption = self._prev_net_consumption[building_idx]
        
        # Ramping = |consumo_actual - consumo_anterior|
        # Normalizado por valor típico (~5 kWh cambio máximo normal)
        ramping = abs(current_consumption - prev_consumption) / 5.0
        
        return float(ramping)
    
    def _get_load_factor_penalty(self, obs: Mapping[str, Union[int, float]]) -> float:
        """
        Calcula penalización por factor de carga bajo.
        
        Load Factor = Consumo promedio / Consumo pico
        Un factor de carga alto (cercano a 1) indica demanda uniforme.
        Un factor de carga bajo indica picos de demanda indeseables.
        
        Se penaliza (consumo^2) para incentivar consumo uniforme (cuadrático).
        """
        # Consumo neto
        net_consumption = float(obs.get('net_electricity_consumption', 0.0))
        
        # Carga no desplazable
        non_shiftable = float(obs.get('non_shiftable_load', 0.0))
        
        # Usar el mayor
        consumption = max(net_consumption, non_shiftable, 0.0)
        
        # Usar consumo al cuadrado para penalizar picos más fuertemente
        # Normalizado por valor típico (10 kWh^2)
        peak_penalty = (consumption ** 2) / 100.0
        
        return peak_penalty
    
    def _get_consumption_penalty(self, obs: Mapping[str, Union[int, float]]) -> float:
        """
        Calcula penalización por consumo eléctrico total.
        
        Incentiva la reducción del consumo total de electricidad,
        promoviendo eficiencia energética general.
        
        Args:
            obs: Observaciones del edificio
            
        Returns:
            Penalización normalizada basada en consumo total
        """
        # Consumo neto de electricidad
        net_consumption = float(obs.get('net_electricity_consumption', 0.0))
        
        # Carga no desplazable (siempre presente)
        non_shiftable = float(obs.get('non_shiftable_load', 0.0))
        
        # Usar el mayor como indicador de consumo real
        consumption = max(net_consumption, non_shiftable)
        
        # Normalizar por consumo típico (~5 kWh promedio)
        consumption_penalty = consumption / 5.0
        
        return consumption_penalty


class CityLearn_5_Metrics_Reward:
    """
    Función de recompensa simplificada que usa las 5 métricas principales
    directamente de las observaciones de CityLearn v2.
    
    Compatible con la estructura de observaciones de CityLearn Challenge 2022.
    """
    
    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        cost_weight: float = 0.20,
        carbon_weight: float = 0.20,
        ramping_weight: float = 0.20,
        peak_weight: float = 0.20,
        consumption_weight: float = 0.20,
        normalize: bool = True
    ):
        """
        Args:
            env_metadata: Metadatos del entorno
            cost_weight: Peso para costo de electricidad
            carbon_weight: Peso para emisiones de CO₂
            ramping_weight: Peso para estabilidad (ramping)
            peak_weight: Peso para control de picos
            consumption_weight: Peso para consumo eléctrico total
            normalize: Si normalizar los valores
        """
        self.env_metadata = env_metadata
        self.central_agent = env_metadata.get('central_agent', False)
        
        # Pesos normalizados
        total = cost_weight + carbon_weight + ramping_weight + peak_weight + consumption_weight
        self.cost_weight = cost_weight / total
        self.carbon_weight = carbon_weight / total
        self.ramping_weight = ramping_weight / total
        self.peak_weight = peak_weight / total
        self.consumption_weight = consumption_weight / total
        self.normalize = normalize
        
        # Tracking
        self._prev_consumption: Optional[np.ndarray] = None
        self._max_values: Mapping[str, float] = {
            'cost': 10.0,
            'carbon': 5.0,
            'ramping': 50.0,
            'peak': 100.0,
            'consumption': 10.0
        }
    
    def reset(self) -> None:
        """Reset al inicio del episodio."""
        self._prev_consumption = None
    
    def calculate(
        self, 
        observations: List[Mapping[str, Union[int, float]]]
    ) -> List[float]:
        """Calcula recompensa basada en 5 métricas principales."""
        n_buildings = len(observations)
        rewards = np.zeros(n_buildings, dtype=np.float32)
        
        # Extraer consumos actuales (usar non_shiftable_load como proxy)
        current_consumption = np.array([
            max(obs.get('net_electricity_consumption', 0.0),
                obs.get('non_shiftable_load', 0.0))
            for obs in observations
        ])
        
        for i, obs in enumerate(observations):
            # 1. COSTO
            consumption = current_consumption[i]
            pricing = obs.get('electricity_pricing', 0.1)
            cost = consumption * pricing
            if self.normalize:
                cost = cost / self._max_values['cost']
            
            # 2. EMISIONES CO₂
            carbon_intensity = obs.get('carbon_intensity', 0.5)
            carbon = consumption * carbon_intensity
            if self.normalize:
                carbon = carbon / self._max_values['carbon']
            
            # 3. RAMPING
            if self._prev_consumption is not None:
                ramping = abs(current_consumption[i] - self._prev_consumption[i])
            else:
                ramping = 0.0
            if self.normalize:
                ramping = ramping / self._max_values['ramping']
            
            # 4. PICO (usando cuadrático)
            peak = consumption ** 2 if consumption > 0 else 0.0
            if self.normalize:
                peak = peak / (self._max_values['peak'] ** 2)
            
            # 5. CONSUMO ELÉCTRICO TOTAL
            elec_consumption = consumption
            if self.normalize:
                elec_consumption = elec_consumption / self._max_values['consumption']
            
            # Combinar con pesos (negativo porque queremos minimizar)
            rewards[i] = -(
                self.cost_weight * cost +
                self.carbon_weight * carbon +
                self.ramping_weight * ramping +
                self.peak_weight * peak +
                self.consumption_weight * elec_consumption
            )
        
        self._prev_consumption = current_consumption.copy()
        
        if self.central_agent:
            return [float(rewards.sum())]
        
        return rewards.tolist()


def create_reward_function(
    env_metadata: Mapping[str, Any],
    reward_type: str = 'maddpg_5_metrics',
    cooperative: bool = True,  # CTDE cooperativo por defecto
    **kwargs
) -> Union[MADDPG_Reward_Function, CityLearn_5_Metrics_Reward]:
    """
    Factory para crear funciones de recompensa CTDE cooperativas.
    
    Args:
        env_metadata: Metadatos del entorno CityLearn
        reward_type: Tipo de función de recompensa
            - 'maddpg_5_metrics': MADDPG cooperativo con 5 métricas
            - 'citylearn_5_metrics': Versión simplificada
        cooperative: Si True, usa team reward (CTDE cooperativo)
        **kwargs: Argumentos adicionales para la función de recompensa
    
    Returns:
        Instancia de función de recompensa cooperativa
    """
    if reward_type in ('maddpg_5_metrics', 'maddpg_4_metrics'):
        return MADDPG_Reward_Function(env_metadata, cooperative=cooperative, **kwargs)
    elif reward_type in ('citylearn_5_metrics', 'citylearn_4_metrics'):
        return CityLearn_5_Metrics_Reward(env_metadata, **kwargs)
    else:
        raise ValueError(f"Tipo de recompensa no soportado: {reward_type}")
