# MADDPG CityLearn - AI Coding Agent Instructions

## Project Overview
**MADDPG (Multi-Agent Deep Deterministic Policy Gradient)** for energy flexibility control in smart building communities. Doctoral thesis implementing **Centralized Training, Decentralized Execution (CTDE)** — 17 building agents coordinate batteries, EVs, HVAC, and solar through a shared Critic.

**Dataset**: `citylearn_challenge_2022_phase_all_plus_evs` — 8760 hourly timesteps (1 year), 17 buildings with EVs, solar PV, batteries, HVAC, DHW.

## Architecture

```
CityLearnMultiAgentEnv (17 agents)
     obs: (17, 28) numpy  →  MADDPG.select_actions()  →  actions: (17, 6) in [-1,1]
                          →  env.step()  →  rewards: (17,)  →  ReplayBuffer  →  update()
```

### Key Source Files (`src/maddpg_tesis/`)
| File | Purpose |
|------|---------|
| `maddpg/maddpg.py` | Orchestrates 17 DDPGAgents, ReplayBuffer. Methods: `select_actions(obs, noise)`, `update()`, `save()` |
| `maddpg/agent.py` | DDPGAgent: Actor (local obs→action) + Critic (global state→Q-value) |
| `envs/citylearn_env.py` | CityLearn wrapper. **ALWAYS use `central_agent=False`** |
| `models/loader.py` | **Only way to load checkpoints**: `load_maddpg(path, device)` — reconstructs from state dict |
| `api/main.py` | FastAPI service with `/predict`, `/health`, `/metrics` endpoints |
| `core/config.py` | Dataclass config from `configs/citylearn_maddpg.yaml` |

## Critical Constraints

| Constraint | Reason |
|------------|--------|
| **CWD must be `maddpg_citylearn/`** | Config paths are relative |
| **`central_agent=False` always** | Multi-agent mode required |
| **Actions in `[-1, 1]`** | CityLearn expects normalized continuous |
| **Use `load_maddpg()` not `torch.load()`** | Checkpoints need reconstruction |
| **Shapes: obs `(n,28)`, actions `(n,6)`, rewards `(n,)`** | Multi-agent parallelism |

## Configuration

**Single source**: `configs/citylearn_maddpg.yaml` — never hardcode hyperparameters.

```python
from maddpg_tesis.core.config import load_config
cfg = load_config()  # → ProjectConfig with cfg.maddpg, cfg.env, cfg.training, cfg.api
```

**Key settings**: `env.reward_weights` (cost/peak/co2/discomfort), `maddpg.update_after=8760` (wait 1 episode), `training.val_every` (validation frequency).

## Developer Workflows

```powershell
# Training (MUST cd first)
cd maddpg_citylearn
python -m maddpg_tesis.scripts.train_citylearn

# Testing (skips model loading)
$env:SKIP_MODEL_LOAD_FOR_TESTS="1"; pytest -q

# API server
uvicorn maddpg_tesis.api.main:app --host 0.0.0.0 --port 8000
```

**Checkpoints saved**: `maddpg.pt` (best train), `maddpg_val_best.pt` (best validation), `maddpg_last.pt`

## CityLearn Installation

CityLearn 2.5.0 has dependency conflicts. Install separately:
```bash
pip install citylearn==2.5.0 --no-deps
pip install gymnasium==0.28.1 pandas "scikit-learn<=1.2.2" simplejson torchvision
```
Automated: `install.ps1`

## Common Errors

| Error | Fix |
|-------|-----|
| `FileNotFoundError: configs/...yaml` | `cd maddpg_citylearn` before running |
| `ValueError: Acciones shape (17, 5)` | Check `env.action_dim` matches model config |
| `ModuleNotFoundError: citylearn` | Run CityLearn install sequence above |
| `RuntimeError: tensors on cuda:0` | Use `core.utils.to_tensor(arr, device)` |
| Actions beyond [-1,1] | Set `noise=False` for inference/validation |

## Device Pattern

```python
from maddpg_tesis.core.utils import get_device, to_tensor
device = get_device("cuda")  # Falls back to CPU
obs_t = to_tensor(obs_np, device)  # Handles dtype/device
```

## Code Style
- Type hints: `np.ndarray`, `torch.Tensor`, `Optional[int]`
- Dataclasses for configs
- Variable names: `obs`, `next_obs`, `rewards` (not `o`, `o2`, `r`)
- After edits: run `pytest -q` and test `/predict` endpoint
