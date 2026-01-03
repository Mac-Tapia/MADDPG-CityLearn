import numpy as np


class GaussianNoiseScheduler:
    """
    Ruido gaussiano con std que decae linealmente desde initial_std a final_std
    en un número dado de pasos (exploration_decay_steps).
    """

    def __init__(
        self,
        action_dim: int,
        initial_std: float = 0.2,
        final_std: float = 0.05,
        decay_steps: int = 500_000,
    ) -> None:
        self.action_dim = action_dim
        self.initial_std = initial_std
        self.final_std = final_std
        self.decay_steps = max(decay_steps, 1)
        self.step_count = 0

    @property
    def current_std(self) -> float:
        frac = min(self.step_count / self.decay_steps, 1.0)
        return self.initial_std + frac * (self.final_std - self.initial_std)

    def step(self) -> None:
        self.step_count += 1

    def sample(self) -> np.ndarray:
        return np.random.normal(
            loc=0.0, scale=self.current_std, size=(self.action_dim,)
        )
