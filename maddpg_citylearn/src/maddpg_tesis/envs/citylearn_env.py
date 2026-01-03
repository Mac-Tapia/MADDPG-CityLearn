from __future__ import annotations

from typing import Any, Dict, Tuple, List, Optional
import numpy as np


class CityLearnMultiAgentEnv:
    """
    Wrapper multi-agente para CityLearn v2.

    - Usa CityLearnEnv con central_agent=False (un agente por edificio).
    - Observaciones: np.ndarray (n_agents, obs_dim)
    - Acciones:      np.ndarray (n_agents, action_dim) en [-1, 1]
    """

    def __init__(
        self, schema: str, central_agent: bool = False, **env_kwargs: Any
    ) -> None:
        """
        Args:
            schema: CityLearn schema name. Ejemplo: 'citylearn_challenge_2022_phase_all_plus_evs'
            central_agent: Debe ser False para multi-agente
            **env_kwargs: Argumentos adicionales para CityLearnEnv
        """
        from citylearn.citylearn import CityLearnEnv

        # Peso de recompensas personalizadas (costos/CO2/comfort/picos)
        self.reward_weights: Optional[Dict[str, float]] = env_kwargs.pop(
            "reward_weights", None
        )

        if central_agent:
            raise ValueError(
                "CityLearnMultiAgentEnv está pensado para MADRL multi-agente, "
                "usa central_agent=False."
            )

        self._env = CityLearnEnv(
            schema=schema, central_agent=False, **env_kwargs
        )

        self.n_agents: int = len(self._env.buildings)

        # Reset inicial para deducir dimensiones (padded a max obs_dim)
        observations, _ = self._env.reset()
        obs_array = self._normalize_obs(observations)
        self.obs_dim: int = obs_array.shape[1]

        self.action_dims = [
            int(space.shape[0]) for space in self._env.action_space
        ]
        self.action_dim: int = max(self.action_dims)

        self._last_obs: np.ndarray = obs_array

    def reset(self) -> np.ndarray:
        """Resetea episodio y devuelve obs (n_agents, obs_dim)."""
        observations, _ = self._env.reset()
        obs_array = self._normalize_obs(observations)

        self._last_obs = obs_array
        return obs_array

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """
        Ejecuta un paso en CityLearn.

        Returns
        -------
        next_obs : (n_agents, obs_dim)
        rewards  : (n_agents,)
        done     : bool
        info     : dict
        """
        actions = np.asarray(actions, dtype=np.float32)

        if actions.shape != (self.n_agents, self.action_dim):
            raise ValueError(
                f"Acciones shape {actions.shape}, esperado "
                f"{(self.n_agents, self.action_dim)}."
            )

        # Ajustar acciones por edificio al número esperado
        per_building_actions = [
            actions[i, : self.action_dims[i]].tolist()
            for i in range(self.n_agents)
        ]

        # Protección contra IndexError al final del episodio (CityLearn bug)
        try:
            next_obs, reward, info, terminated, truncated = self._env.step(
                per_building_actions
            )
        except IndexError:
            # Fin del episodio - devolver última observación con done=True
            return (
                self._last_obs,
                np.zeros(self.n_agents, dtype=np.float32),
                True,
                {"truncated_by_index": True},
            )

        info_dict: Dict[str, Any] = info if isinstance(info, dict) else {}

        next_obs_array = self._normalize_obs(next_obs)
        rewards_array = np.asarray(reward, dtype=np.float32)

        if rewards_array.ndim == 0:
            rewards_array = np.full(
                self.n_agents, rewards_array, dtype=np.float32
            )
        elif rewards_array.shape != (self.n_agents,):
            raise RuntimeError(
                f"reward inesperado: esperado (n_agents,), recibido {rewards_array.shape}."
            )

        # Recompensa personalizada: pondera costos/CO2/disconfort/picos si se proveen pesos
        if self.reward_weights:
            rewards_array = self._custom_reward(rewards_array, info_dict)

        done: bool = bool(terminated or truncated)
        self._last_obs = next_obs_array
        return next_obs_array, rewards_array, done, info_dict

    def evaluate(self):
        """Devuelve KPIs normalizados de CityLearn."""
        return self._env.evaluate()

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    # ----------------------------------------------------
    # Helpers
    # ----------------------------------------------------
    def _normalize_obs(self, obs_list: List[Any]) -> np.ndarray:
        """Convierte lista de obs de longitudes variables a matriz padded."""
        if len(obs_list) != self.n_agents:
            raise RuntimeError(
                f"Observaciones con n_agents inesperado: {len(obs_list)} != {self.n_agents}"
            )

        max_dim = max(len(o) for o in obs_list)
        arr = np.zeros((self.n_agents, max_dim), dtype=np.float32)
        for i, o in enumerate(obs_list):
            o_arr = np.asarray(o, dtype=np.float32)
            arr[i, : len(o_arr)] = o_arr
        return arr

    def _custom_reward(
        self, base_reward: np.ndarray, info: Dict[str, Any]
    ) -> np.ndarray:
        """Crea recompensa ponderada a partir de métricas de CityLearn.

        Usa claves comunes si existen en info:
        - costos: electricity_costs | costs | total_cost
        - emisiones: carbon_emissions | emissions
        - disconfort: discomfort
        - picos: peak_demand | peak_net_electricity_consumption
        """

        def get_arr(keys, default=0.0):
            for k in keys:
                if k in info:
                    v = info[k]
                    arr = np.asarray(v, dtype=np.float32)
                    if arr.ndim == 0:
                        arr = np.full(self.n_agents, arr, dtype=np.float32)
                    return arr
            return np.full(self.n_agents, default, dtype=np.float32)

        w = self.reward_weights or {}
        cost = get_arr(["electricity_costs", "costs", "total_cost"], 0.0)
        co2 = get_arr(["carbon_emissions", "emissions"], 0.0)
        discomfort = get_arr(["discomfort"], 0.0)
        peak = get_arr(
            ["peak_demand", "peak_net_electricity_consumption"], 0.0
        )

        # La recompensa es negativa de costos/penalizaciones (para maximizar reward)
        custom = base_reward.copy()
        custom -= w.get("cost", 0.0) * cost
        custom -= w.get("co2", 0.0) * co2
        custom -= w.get("discomfort", 0.0) * discomfort
        custom -= w.get("peak", 0.0) * peak
        return custom
