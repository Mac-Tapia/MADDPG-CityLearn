"""
Tests para los componentes core del proyecto
"""
import pytest
import numpy as np


def test_config_loading():
    """Test loading configuration"""
    from maddpg_tesis.core.config import (
        ProjectConfig,
        MADDPGConfig,
        EnvConfig,
        TrainingConfig,
        APIConfig,
    )

    # Test default configs
    env_cfg = EnvConfig()
    assert env_cfg.schema == "citylearn_challenge_2022_phase_all_plus_evs"

    maddpg_cfg = MADDPGConfig()
    assert maddpg_cfg.gamma == 0.99
    assert maddpg_cfg.tau == 0.005

    training_cfg = TrainingConfig()
    assert training_cfg.episodes == 50

    api_cfg = APIConfig()
    assert api_cfg.host == "0.0.0.0"
    assert api_cfg.port == 8000


def test_device_detection():
    """Test device detection utility"""
    from maddpg_tesis.core.utils import get_device
    import torch

    device = get_device("cpu")
    assert device == torch.device("cpu")

    # Test CUDA si está disponible
    if torch.cuda.is_available():
        device = get_device("cuda")
        assert device == torch.device("cuda")


def test_seed_setting():
    """Test global seed setting"""
    from maddpg_tesis.core.utils import set_global_seeds
    import random
    import torch

    set_global_seeds(42)

    # Verificar que los seeds están configurados
    r1 = random.random()
    set_global_seeds(42)
    r2 = random.random()
    assert r1 == r2

    t1 = torch.rand(1).item()
    set_global_seeds(42)
    torch.rand(1)  # skip one
    t2 = torch.rand(1).item()
    assert t1 != t2  # Different after resetting
