"""
Script para imprimir claves de info y dimensiones de obs/acciones en CityLearn.
Uso:
  python scripts/debug_info_keys.py
"""
import pprint
import numpy as np

from maddpg_tesis.envs.citylearn_env import CityLearnMultiAgentEnv


def main():
    env = CityLearnMultiAgentEnv(
        schema="citylearn_challenge_2022_phase_all_plus_evs"
    )
    print(
        f"n_agents={env.n_agents}, obs_dim={env.obs_dim}, action_dim={env.action_dim}"
    )

    obs = env.reset()
    print(f"obs shape: {obs.shape}")

    actions = env._env.action_space
    print("action space por agente:")
    for i, space in enumerate(actions):
        print(f"  agent {i+1}: shape={space.shape}")

    # Step con acciones aleatorias del espacio original, padded a (n_agents, max_dim)
    max_dim = max(space.shape[0] for space in actions)
    padded_actions = []
    for i, space in enumerate(actions):
        a = space.sample()
        pad = np.zeros(max_dim, dtype=np.float32)
        pad[: a.shape[0]] = a
        padded_actions.append(pad)

    next_obs, rewards, done, info = env.step(np.stack(padded_actions, axis=0))

    print(f"next_obs shape: {next_obs.shape}")
    print(
        f"rewards shape: {rewards.shape}, sample={rewards[:min(3, len(rewards))]}"
    )
    print(f"done: {done}")

    print("\nClaves en info:")
    if isinstance(info, dict):
        pprint.pp(list(info.keys()))

        # Mostrar algunas claves típicas si existen
        for key in [
            "electricity_costs",
            "carbon_emissions",
            "discomfort",
            "peak_demand",
            "peak_net_electricity_consumption",
        ]:
            if key in info:
                print(
                    f"{key}: shape={getattr(info[key], 'shape', 'scalar')} | sample={info[key]}"
                )
    else:
        print(f"info no es dict, es {type(info)} -> {info}")

    env.close()


if __name__ == "__main__":
    main()
