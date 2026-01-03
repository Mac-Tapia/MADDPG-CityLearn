import numpy as np


class OrnsteinUhlenbeckNoise:
    """
    Ruido Ornstein-Uhlenbeck para exploración continua en DDPG/MADDPG.
    
    Este ruido tiene correlación temporal, lo que produce trayectorias
    de exploración más suaves y eficientes que ruido Gaussiano puro.
    
    dx = theta * (mu - x) * dt + sigma * dW
    
    VENTAJA sobre ruido Gaussiano:
    - Exploración más coherente (acciones similares en timesteps consecutivos)
    - Mejor para tareas de control continuo como gestión energética
    - MARLISA usa decaimiento similar, pero OU es más sofisticado
    """
    
    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        sigma_min: float = 0.02,
        sigma_decay: float = 0.9995,
    ) -> None:
        """
        Args:
            action_dim: Dimensión del espacio de acciones
            mu: Media de reversión (típicamente 0)
            theta: Velocidad de reversión a la media (mayor = más rápido)
            sigma: Volatilidad inicial (exploración)
            sigma_min: Volatilidad mínima
            sigma_decay: Factor de decaimiento de sigma por step
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.state = np.ones(action_dim) * mu
        self._initial_sigma = sigma
        
    def reset(self) -> None:
        """Reset al estado inicial (llamar al inicio de cada episodio)."""
        self.state = np.ones(self.action_dim) * self.mu
        
    def sample(self) -> np.ndarray:
        """Genera ruido OU correlacionado temporalmente."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state
    
    def step(self) -> None:
        """Decay del sigma (llamar cada step de entrenamiento)."""
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)
    
    @property
    def current_std(self) -> float:
        """Compatibilidad con interfaz anterior."""
        return self.sigma


class GaussianNoiseScheduler:
    """
    Ruido gaussiano con std que decae linealmente desde initial_std a final_std
    en un número dado de pasos (exploration_decay_steps).
    
    NOTA: OrnsteinUhlenbeckNoise es preferible para MADDPG/DDPG.
    Esta clase se mantiene por compatibilidad.
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
    
    def reset(self) -> None:
        """Reset (no-op para Gaussiano, pero necesario para interfaz unificada)."""
        pass
