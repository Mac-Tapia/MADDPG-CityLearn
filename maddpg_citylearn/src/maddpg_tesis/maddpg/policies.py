from typing import Optional, Sequence

import torch
import torch.nn as nn


def mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: nn.Module = nn.ReLU(),
    output_activation: Optional[nn.Module] = None,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation)
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Actor profundo para acciones continuas en [-1, 1]."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.net = mlp(
            input_dim=obs_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=action_dim,
            activation=nn.ReLU(),
            output_activation=nn.Tanh(),
            layer_norm=layer_norm,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    """
    Crítico centralizado profundo:
    entrada = [obs_global, actions_global].
    """

    def __init__(
        self,
        global_obs_dim: int,
        global_action_dim: int,
        hidden_dim: int,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        input_dim = global_obs_dim + global_action_dim
        self.net = mlp(
            input_dim=input_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=1,
            activation=nn.ReLU(),
            output_activation=None,
            layer_norm=layer_norm,
        )

    def forward(
        self, global_obs: torch.Tensor, global_actions: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([global_obs, global_actions], dim=-1)
        return self.net(x)
