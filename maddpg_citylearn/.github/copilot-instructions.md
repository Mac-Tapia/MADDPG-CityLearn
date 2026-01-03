# MADDPG CityLearn - AI Coding Agent Instructions

## Project Overview

**MADDPG (Multi-Agent Deep Deterministic Policy Gradient)** for energy flexibility control in smart building communities. Doctoral thesis implementing **Centralized Training, Decentralized Execution (CTDE)** — 17 building agents coordinate batteries, EVs, HVAC, and solar through a shared Critic.

**Dataset**: `citylearn_challenge_2022_phase_all_plus_evs` — 8760 hourly timesteps (1 year), 17 buildings with EVs, solar PV, batteries, HVAC, DHW.

## Architecture

```
CityLearnMultiAgentEnv (17 agents)
    obs: (17, 28) numpy → MADDPG.select_actions() → actions: (17, 6) in [-1,1]
                        → env.step() → rewards: (17,) → ReplayBuffer → update()
```

### Source Structure (`src/maddpg_tesis/`)

| Path | Purpose |
|------|---------|
| `maddpg/maddpg.py` | Orchestrates 17 DDPGAgents, ReplayBuffer. Key: `select_actions(obs, noise)`, `update()`, `save()` |
| `maddpg/agent.py` | DDPGAgent with Actor (local obs→action) + Critic (global state→Q-value) |
| `maddpg/policies.py` | Neural networks (Actor uses tanh for [-1,1] bounds) |
| `envs/citylearn_env.py` | CityLearn wrapper. **MUST use `central_agent=False`** |
| `models/loader.py` | **Only way to load checkpoints**: `load_maddpg(path, device)` |
| `api/main.py` | FastAPI: `/predict`, `/health`, `/metrics`, `/simulate/step` |
| `core/config.py` | Dataclass config parsed from YAML |

## Critical Constraints

| Constraint | Why |
|------------|-----|
| **CWD = `maddpg_citylearn/`** | Config paths `configs/*.yaml` are relative |
| **`central_agent=False`** | Multi-agent mode required for MADDPG |
| **Actions ∈ `[-1, 1]`** | CityLearn expects normalized continuous actions |
| **Use `load_maddpg()`** | Never `torch.load()` directly — needs reconstruction |
| **Shapes: `(n,28)`, `(n,6)`, `(n,)`** | n=17 agents, obs/action/reward dimensions |

## Configuration (Single Source of Truth)

All hyperparameters live in `configs/citylearn_maddpg.yaml`. Never hardcode.

```python
from maddpg_tesis.core.config import load_config
cfg = load_config()  # ProjectConfig with cfg.maddpg, cfg.env, cfg.training, cfg.api
```

Key settings to understand:
- `env.reward_weights`: dict with `cost`, `peak`, `co2`, `discomfort` weights
- `maddpg.update_after`: 8760 (wait 1 full episode before updates)
- `maddpg.device`: `"cuda"` or `"cpu"`
- `training.val_every`: validation frequency (episodes)
- `training.early_stopping_patience`: epochs without improvement

## Developer Workflows

```powershell
# 1. Training (MUST cd first, set PYTHONPATH)
cd maddpg_citylearn
$env:PYTHONPATH="src"; python -u scripts/train_citylearn.py

# 2. Testing (skips slow model loading)
$env:SKIP_MODEL_LOAD_FOR_TESTS="1"; pytest -q

# 3. API server
cd maddpg_citylearn
uvicorn maddpg_tesis.api.main:app --host 0.0.0.0 --port 8000

# 4. Docker
docker build -t maddpg-citylearn .
docker run -p 8000:8000 -v $(pwd)/models:/app/models maddpg-citylearn
```

**Checkpoints saved to `models/citylearn_maddpg/`**:
- `maddpg.pt` — best training reward
- `maddpg_val_best.pt` — best validation reward
- `maddpg_last.pt` — latest state (for resume)

## CityLearn Installation (Dependency Conflicts)

CityLearn 2.5.0 conflicts with modern packages. Install separately:

```bash
pip install citylearn==2.5.0 --no-deps
pip install gymnasium==0.28.1 pandas "scikit-learn<=1.2.2" simplejson torchvision
```

Or run `install.ps1` for automated setup.

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `FileNotFoundError: configs/...yaml` | Run from `maddpg_citylearn/` directory |
| `ValueError: Acciones shape (17, 5)` | Mismatch action_dim — check env vs model |
| `ModuleNotFoundError: citylearn` | Run CityLearn install sequence |
| `RuntimeError: tensors on cuda:0` | Use `to_tensor(arr, device)` from `core.utils` |
| Actions beyond [-1,1] during inference | Set `noise=False` in `select_actions()` |
| `IndexError` at episode end | CityLearn bug — wrapper handles gracefully |

## Code Patterns

### Device-Agnostic Tensors
```python
from maddpg_tesis.core.utils import get_device, to_tensor
device = get_device("cuda")  # Falls back to CPU if unavailable
obs_t = to_tensor(obs_np, device)  # Handles dtype/device conversion
```

### Loading Checkpoints (ALWAYS use this)
```python
from maddpg_tesis.models.loader import load_maddpg
maddpg = load_maddpg("models/citylearn_maddpg/maddpg.pt", device="cuda")
```

### Creating Environment
```python
from maddpg_tesis.envs import CityLearnMultiAgentEnv
env = CityLearnMultiAgentEnv(
    schema="citylearn_challenge_2022_phase_all_plus_evs",
    central_agent=False,  # REQUIRED for MADDPG
    random_seed=42
)
```

## Style & Conventions

- **Type hints required**: `np.ndarray`, `torch.Tensor`, `Optional[int]`
- **Dataclasses for configs**: see `core/config.py` pattern
- **Variable naming**: `obs`, `next_obs`, `rewards`, `actions` (not `o`, `o2`, `r`, `a`)
- **After edits**: run `pytest -q` and test `/predict` endpoint
- **Spanish comments allowed** (thesis context)

## Testing

Tests in `tests/` use `SKIP_MODEL_LOAD_FOR_TESTS=1` to bypass slow checkpoint loading:

```python
# tests/conftest.py sets this automatically
os.environ.setdefault("SKIP_MODEL_LOAD_FOR_TESTS", "1")
```

Key test files: `test_maddpg.py` (Actor/Critic/ReplayBuffer), `test_api.py` (endpoints), `test_core.py` (utils).
