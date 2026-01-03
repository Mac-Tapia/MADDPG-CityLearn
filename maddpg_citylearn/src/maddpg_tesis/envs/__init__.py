"""Wrappers de entornos (CityLearn v2, etc.)."""
"""
Wrappers de entornos para MADRL.

Incluye función de recompensa personalizada para las 5 métricas principales
de CityLearn v2: Cost, Carbon Emissions, Ramping, Load Factor, Electricity Consumption.
"""

from .citylearn_env import CityLearnMultiAgentEnv
from .reward_functions import (
    MADDPG_Reward_Function,
    CityLearn_5_Metrics_Reward,
    create_reward_function,
)

__all__ = [
    "CityLearnMultiAgentEnv",
    "MADDPG_Reward_Function",
    "CityLearn_5_Metrics_Reward",
    "create_reward_function",
]
