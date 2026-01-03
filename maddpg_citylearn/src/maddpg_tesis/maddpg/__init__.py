"""Implementación de MADDPG para MADRL."""
"""
Implementación de MADDPG para entornos de Multi-Agent Deep RL (MADRL).

Incluye:
- MADDPG: Implementación básica
- CooperativeMADDPG: MADDPG con coordinación explícita entre agentes
- CooperativeCoordinator: Módulo de coordinación (Mean-Field + Attention)
"""

from .maddpg import MADDPG
from .cooperative_maddpg import CooperativeMADDPG
from .coordination import (
    CooperativeCoordinator,
    create_coordination_module,
    DistrictAggregator,
    AttentionCoordination,
    MeanFieldModule,
)

__all__ = [
    "MADDPG",
    "CooperativeMADDPG",
    "CooperativeCoordinator",
    "create_coordination_module",
    "DistrictAggregator",
    "AttentionCoordination",
    "MeanFieldModule",
]
