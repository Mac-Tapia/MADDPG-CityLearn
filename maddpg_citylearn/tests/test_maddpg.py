"""
Tests para los componentes MADDPG
"""
import pytest
import torch
import numpy as np


def test_actor_network():
    """Test Actor network initialization and forward pass"""
    from maddpg_tesis.maddpg.policies import Actor

    obs_dim = 10
    action_dim = 3
    hidden_dim = 64

    actor = Actor(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim
    )

    # Test forward pass
    obs = torch.randn(4, obs_dim)  # batch_size=4
    actions = actor(obs)

    assert actions.shape == (4, action_dim)
    # Check tanh bounds
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)


def test_critic_network():
    """Test Critic network initialization and forward pass"""
    from maddpg_tesis.maddpg.policies import Critic

    n_agents = 3
    obs_dim = 10
    action_dim = 3
    hidden_dim = 64

    global_obs_dim = n_agents * obs_dim
    global_action_dim = n_agents * action_dim

    critic = Critic(
        global_obs_dim=global_obs_dim,
        global_action_dim=global_action_dim,
        hidden_dim=hidden_dim,
    )

    # Test forward pass
    batch_size = 4
    global_obs = torch.randn(batch_size, global_obs_dim)
    global_actions = torch.randn(batch_size, global_action_dim)

    q_values = critic(global_obs, global_actions)

    assert q_values.shape == (batch_size, 1)


def test_replay_buffer():
    """Test ReplayBuffer operations"""
    from maddpg_tesis.maddpg.replay_buffer import ReplayBuffer

    capacity = 100
    n_agents = 3
    obs_dim = 10
    action_dim = 3

    buffer = ReplayBuffer(
        capacity=capacity,
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    # Add transitions
    for _ in range(10):
        obs = np.random.randn(n_agents, obs_dim)
        actions = np.random.randn(n_agents, action_dim)
        rewards = np.random.randn(n_agents)
        next_obs = np.random.randn(n_agents, obs_dim)
        dones = np.zeros(n_agents)

        buffer.add(obs, actions, rewards, next_obs, dones)

    assert len(buffer) == 10

    # Sample batch
    batch = buffer.sample(batch_size=5)

    assert batch["obs"].shape == (5, n_agents, obs_dim)
    assert batch["actions"].shape == (5, n_agents, action_dim)
    assert batch["rewards"].shape == (5, n_agents)
    assert batch["next_obs"].shape == (5, n_agents, obs_dim)
    assert batch["dones"].shape == (5, n_agents)


def test_gaussian_noise_scheduler():
    """Test Gaussian noise scheduler"""
    from maddpg_tesis.maddpg.noise import GaussianNoiseScheduler

    action_dim = 3
    initial_std = 0.2
    final_std = 0.05
    decay_steps = 1000

    noise_scheduler = GaussianNoiseScheduler(
        action_dim=action_dim,
        initial_std=initial_std,
        final_std=final_std,
        decay_steps=decay_steps,
    )

    # Check initial std
    assert noise_scheduler.current_std == initial_std

    # Step through half decay
    for _ in range(500):
        noise_scheduler.step()

    mid_std = noise_scheduler.current_std
    assert initial_std > mid_std > final_std

    # Step to end
    for _ in range(500):
        noise_scheduler.step()

    # Should be at or near final_std
    assert abs(noise_scheduler.current_std - final_std) < 0.01

    # Sample noise
    noise = noise_scheduler.sample()
    assert noise.shape == (action_dim,)


@pytest.mark.skip(reason="Requiere CityLearn instalado correctamente")
def test_citylearn_env_wrapper():
    """Test CityLearn environment wrapper"""
    from maddpg_tesis.envs.citylearn_env import CityLearnMultiAgentEnv

    env = CityLearnMultiAgentEnv(
        schema="citylearn_challenge_2023_phase_2_local_evaluation",
        central_agent=False,
    )

    assert env.n_agents > 0
    assert env.obs_dim > 0
    assert env.action_dim > 0

    # Test reset
    obs = env.reset()
    assert obs.shape == (env.n_agents, env.obs_dim)

    # Test step
    actions = np.random.uniform(-1, 1, (env.n_agents, env.action_dim))
    next_obs, rewards, done, info = env.step(actions)

    assert next_obs.shape == (env.n_agents, env.obs_dim)
    assert rewards.shape == (env.n_agents,)
    assert isinstance(done, bool)
