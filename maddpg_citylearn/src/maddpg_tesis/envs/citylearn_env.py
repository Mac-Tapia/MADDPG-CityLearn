from __future__ import annotations

from typing import Any, Dict, Tuple, List, Optional
import warnings
import numpy as np


class CityLearnMultiAgentEnv:
    """
    Wrapper multi-agente robusto para CityLearn v2.

    - Usa CityLearnEnv con central_agent=False (un agente por edificio).
    - Observaciones: np.ndarray (n_agents, obs_dim)
    - Acciones:      np.ndarray (n_agents, action_dim) en [-1, 1]
    - Manejo robusto de errores y excepciones de CityLearn.
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
        # Suprimir warnings de CityLearn durante inicialización
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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

        # Inicializar CityLearn con manejo de errores
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._env = CityLearnEnv(
                    schema=schema, central_agent=False, **env_kwargs
                )
        except Exception as e:
            raise RuntimeError(f"Error inicializando CityLearn: {e}") from e

        self.n_agents: int = len(self._env.buildings)
        self._is_closed: bool = False
        self._episode_step: int = 0
        self._max_episode_steps: int = 8760  # Default: 1 año de simulación

        # Reset inicial para deducir dimensiones (padded a max obs_dim)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                observations, _ = self._env.reset()
            obs_array = self._normalize_obs(observations)
            self.obs_dim: int = obs_array.shape[1]
        except Exception as e:
            raise RuntimeError(f"Error en reset inicial: {e}") from e

        self.action_dims = [
            int(space.shape[0]) for space in self._env.action_space
        ]
        self.action_dim: int = max(self.action_dims)

        self._last_obs: np.ndarray = obs_array

    def reset(self) -> np.ndarray:
        """Resetea episodio y devuelve obs (n_agents, obs_dim)."""
        if self._is_closed:
            raise RuntimeError("El entorno está cerrado, no se puede resetear")
        
        self._episode_step = 0
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                observations, _ = self._env.reset()
            obs_array = self._normalize_obs(observations)
        except Exception as e:
            # Intentar recuperarse recreando el entorno interno
            raise RuntimeError(f"Error en reset: {e}") from e

        self._last_obs = obs_array
        return obs_array

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """
        Ejecuta un paso en CityLearn con manejo robusto de errores.

        Returns
        -------
        next_obs : (n_agents, obs_dim)
        rewards  : (n_agents,)
        done     : bool
        info     : dict
        """
        if self._is_closed:
            return (
                self._last_obs,
                np.zeros(self.n_agents, dtype=np.float32),
                True,
                {"error": "environment_closed"},
            )
        
        actions = np.asarray(actions, dtype=np.float32)
        
        # Validar y sanitizar acciones
        if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
            actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = np.clip(actions, -1.0, 1.0)

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

        # Protección contra excepciones de CityLearn
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                next_obs, reward, info, terminated, truncated = self._env.step(
                    per_building_actions
                )
            self._episode_step += 1
        except IndexError:
            # Fin del episodio - bug conocido de CityLearn
            return (
                self._last_obs,
                np.zeros(self.n_agents, dtype=np.float32),
                True,
                {"truncated_by_index": True},
            )
        except KeyError as e:
            # Error de acceso a datos
            return (
                self._last_obs,
                np.zeros(self.n_agents, dtype=np.float32),
                True,
                {"error": f"key_error: {e}"},
            )
        except Exception as e:
            # Otros errores - terminar episodio gracefully
            return (
                self._last_obs,
                np.zeros(self.n_agents, dtype=np.float32),
                True,
                {"error": str(e)},
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
        if self._is_closed:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self._env.evaluate()
        except Exception:
            return None

    def close(self) -> None:
        """Cierra el entorno de forma segura."""
        if self._is_closed:
            return
        self._is_closed = True
        try:
            if hasattr(self._env, "close"):
                self._env.close()
        except Exception:
            pass

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
