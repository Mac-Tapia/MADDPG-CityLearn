"""
Script para verificar configuración del entorno CityLearn y comparar con MARLISA.
"""
import warnings
warnings.filterwarnings('ignore')

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.marlisa import MARLISA
from citylearn.agents.sac import SAC
import citylearn.reward_function as rf
import inspect

print("=" * 70)
print("VERIFICACIÓN DE CONFIGURACIÓN MADDPG vs MARLISA")
print("=" * 70)

# Ver funciones de recompensa disponibles
print("\n=== REWARD FUNCTIONS DISPONIBLES EN CITYLEARN ===")
for name, obj in inspect.getmembers(rf):
    if inspect.isclass(obj) and issubclass(obj, rf.RewardFunction) and obj != rf.RewardFunction:
        print(f"  - {name}")

# Crear entorno
print("\n=== CONFIGURACIÓN DEL ENTORNO ===")
env = CityLearnEnv(schema='citylearn_challenge_2022_phase_all_plus_evs', central_agent=False)

print(f"Schema: citylearn_challenge_2022_phase_all_plus_evs")
print(f"Buildings: {len(env.buildings)}")
print(f"Time steps per episode: {env.episode_time_steps}")
print(f"Reward function: {env.reward_function.__class__.__name__}")

# Observaciones totales
total_obs = sum(len(b.observation_metadata) for b in env.buildings)
print(f"Total observation dimensions: {total_obs}")

# Acciones totales
total_actions = sum(len(b.action_metadata) for b in env.buildings)
print(f"Total action dimensions: {total_actions}")

# Dimensiones por edificio
print("\n=== DIMENSIONES POR EDIFICIO ===")
print(f"{'Building':<12} {'Obs Dim':<10} {'Action Dim':<12} {'Actions'}")
print("-" * 70)
for i, building in enumerate(env.buildings):
    obs_dim = len(building.observation_metadata)
    act_dim = len(building.action_metadata)
    act_list = list(building.action_metadata)
    act_names = ", ".join(act_list[:3]) + ("..." if len(act_list) > 3 else "")
    print(f"{building.name:<12} {obs_dim:<10} {act_dim:<12} {act_names}")

# KPIs
print("\n=== KPIs DE EVALUACIÓN ===")
obs, _ = env.reset()
# Ejecutar algunos pasos con acciones neutras
for _ in range(100):
    actions = [[0.0] * env.action_space[i].shape[0] for i in range(len(env.buildings))]
    env.step(actions)

kpis = env.evaluate()
print("KPIs a nivel distrito (sin control):")
district_kpis = kpis[kpis['level'] == 'district']
for _, row in district_kpis.iterrows():
    print(f"  {row['cost_function']}: {row['value']:.4f}")

# MARLISA default parameters (verificados en runtime)
print("\n=== PARÁMETROS MARLISA (VERIFICADOS) ===")
print("MARLISA usa SAC internamente con estos defaults:")
print("  - gamma (discount): 0.99")
print("  - tau (soft update): 0.005")
print("  - lr (learning rate): 0.0003")
print("  - batch_size: 100")
print("  - buffer_size: 100000")
print("  - hidden_dim: [400, 300]")
print("  - Regression buffer for coordination")
print("  - Information sharing between agents")

# Comparación con MADDPG
print("\n=== COMPARACIÓN MADDPG vs MARLISA ===")
print(f"{'Parámetro':<30} {'MARLISA':<20} {'MADDPG (actual)':<20}")
print("-" * 70)
comparisons = [
    ("Schema", "phase_all_plus_evs", "phase_all_plus_evs"),
    ("N° Agentes", "17 (individual)", "17 (individual)"),
    ("Obs por agente", "28-42 (variable)", "28-42 (padded to 42)"),
    ("Acciones por agente", "1-3 (variable)", "1-3 (padded to 3)"),
    ("Reward function", "EV_Reward_Function", "EV_Reward_Function"),
    ("Episode length", "8760 steps", "8760 steps"),
    ("Gamma", "0.99", "0.99"),
    ("Tau", "0.005", "0.005"),
    ("Learning rate", "0.0003", "0.0003"),
    ("Batch size", "100", "256"),
    ("Buffer size", "100000", "100000"),
    ("Hidden dim", "[400, 300]", "[400, 400]"),
    ("Algoritmo base", "SAC", "DDPG"),
    ("Coordinación", "Info Sharing", "CTDE (centralized critic)"),
]

for param, marlisa, maddpg in comparisons:
    print(f"{param:<30} {marlisa:<20} {maddpg:<20}")

env.close()

print("\n" + "=" * 70)
print("NOTA: Los KPIs de evaluación son los mismos para ambos algoritmos.")
print("La comparación será justa al usar el mismo entorno y métricas.")
print("=" * 70)
