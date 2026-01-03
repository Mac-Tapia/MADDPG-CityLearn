"""
Script para evaluar MARLISA y baselines en CityLearn v2.5.
Genera resultados comparables con MADDPG para la tesis.

Dataset: citylearn_challenge_2022_phase_all_plus_evs (17 edificios + EVs)
Métricas: KPIs de flexibilidad energética normalizados
"""
import json
import warnings
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

warnings.filterwarnings('ignore')

# Añadir src al path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from citylearn.citylearn import CityLearnEnv
from citylearn.agents.marlisa import MARLISA
from citylearn.agents.sac import SAC
from citylearn.agents.rbc import BasicRBC, OptimizedRBC

# Configuración
SCHEMA = "citylearn_challenge_2022_phase_all_plus_evs"
OUTPUT_DIR = ROOT / "models" / "baselines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_agent(env: CityLearnEnv, agent, agent_name: str, episodes: int = 1) -> Dict[str, Any]:
    """Evalúa un agente y retorna métricas."""
    print(f"\n{'='*60}")
    print(f"Evaluando: {agent_name}")
    print(f"{'='*60}")
    
    all_rewards = []
    
    for ep in range(episodes):
        observations, _ = env.reset()
        agent.reset()
        
        episode_reward = 0.0
        steps = 0
        done = False
        
        while not done:
            # Obtener acciones del agente
            actions = agent.select_actions(observations)
            
            # Ejecutar step
            observations, reward, info, terminated, truncated = env.step(actions)
            done = terminated or truncated
            
            # Acumular reward
            if isinstance(reward, list):
                episode_reward += sum(reward)
            else:
                episode_reward += reward
            
            steps += 1
            
            if steps % 1000 == 0:
                print(f"  Step {steps}/8760")
        
        all_rewards.append(episode_reward)
        print(f"  Episodio {ep+1}: reward={episode_reward:.2f}, steps={steps}")
    
    # Obtener KPIs
    kpis = env.evaluate()
    
    # Extraer KPIs de distrito
    district_kpis = {}
    if hasattr(kpis, 'iterrows'):
        for _, row in kpis[kpis['level'] == 'district'].iterrows():
            value = row['value']
            if not np.isnan(value):
                district_kpis[row['cost_function']] = float(value)
    
    return {
        "agent": agent_name,
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)) if len(all_rewards) > 1 else 0.0,
        "episodes": episodes,
        "kpis": district_kpis
    }


def run_no_control_baseline(episodes: int = 1) -> Dict[str, Any]:
    """Ejecuta baseline sin control (acciones = 0)."""
    print("\n" + "="*60)
    print("Evaluando: No Control (Baseline)")
    print("="*60)
    
    env = CityLearnEnv(schema=SCHEMA, central_agent=False)
    
    all_rewards = []
    
    for ep in range(episodes):
        observations, _ = env.reset()
        episode_reward = 0.0
        steps = 0
        done = False
        
        while not done:
            # Acciones neutras (sin control)
            actions = [[0.0] * env.action_space[i].shape[0] for i in range(len(env.buildings))]
            observations, reward, info, terminated, truncated = env.step(actions)
            done = terminated or truncated
            
            if isinstance(reward, list):
                episode_reward += sum(reward)
            else:
                episode_reward += reward
            
            steps += 1
            
            if steps % 2000 == 0:
                print(f"  Step {steps}/8760")
        
        all_rewards.append(episode_reward)
        print(f"  Episodio {ep+1}: reward={episode_reward:.2f}, steps={steps}")
    
    kpis = env.evaluate()
    district_kpis = {}
    if hasattr(kpis, 'iterrows'):
        for _, row in kpis[kpis['level'] == 'district'].iterrows():
            value = row['value']
            if not np.isnan(value):
                district_kpis[row['cost_function']] = float(value)
    
    env.close()
    
    return {
        "agent": "No Control (Baseline)",
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)) if len(all_rewards) > 1 else 0.0,
        "episodes": episodes,
        "kpis": district_kpis
    }


def run_rbc_baseline(episodes: int = 1) -> Dict[str, Any]:
    """Ejecuta Rule-Based Controller."""
    print("\n" + "="*60)
    print("Evaluando: RBC (Rule-Based Control)")
    print("="*60)
    
    env = CityLearnEnv(schema=SCHEMA, central_agent=False)
    
    try:
        agent = OptimizedRBC(env)
    except Exception:
        try:
            agent = BasicRBC(env)
        except Exception as e:
            print(f"  [!] No se pudo crear RBC: {e}")
            env.close()
            return None
    
    result = evaluate_agent(env, agent, "RBC", episodes)
    env.close()
    return result


def run_marlisa(episodes: int = 1) -> Dict[str, Any]:
    """Ejecuta MARLISA (el benchmark principal)."""
    print("\n" + "="*60)
    print("Evaluando: MARLISA (Multi-Agent RL + Information Sharing)")
    print("="*60)
    
    env = CityLearnEnv(schema=SCHEMA, central_agent=False)
    
    try:
        agent = MARLISA(env)
        print("  [+] MARLISA inicializado correctamente")
    except Exception as e:
        print(f"  [!] Error inicializando MARLISA: {e}")
        env.close()
        return None
    
    result = evaluate_agent(env, agent, "MARLISA", episodes)
    env.close()
    return result


def run_sac_independent(episodes: int = 1) -> Dict[str, Any]:
    """Ejecuta SAC independiente (sin coordinación)."""
    print("\n" + "="*60)
    print("Evaluando: SAC Independiente")
    print("="*60)
    
    env = CityLearnEnv(schema=SCHEMA, central_agent=False)
    
    try:
        agent = SAC(env)
        print("  [+] SAC inicializado correctamente")
    except Exception as e:
        print(f"  [!] Error inicializando SAC: {e}")
        env.close()
        return None
    
    result = evaluate_agent(env, agent, "SAC (Independent)", episodes)
    env.close()
    return result


def main():
    """Función principal."""
    print("="*70)
    print("EVALUACIÓN DE BASELINES PARA COMPARACIÓN CON MADDPG")
    print("Dataset: citylearn_challenge_2022_phase_all_plus_evs")
    print("="*70)
    
    results = []
    
    # 1. No Control (baseline obligatorio)
    result = run_no_control_baseline(episodes=1)
    if result:
        results.append(result)
    
    # 2. RBC (si está disponible)
    result = run_rbc_baseline(episodes=1)
    if result:
        results.append(result)
    
    # 3. MARLISA (benchmark principal)
    # Nota: MARLISA requiere entrenamiento previo
    # Para evaluación rápida, usamos pocos episodios
    print("\n[!] NOTA: MARLISA requiere entrenamiento. Saltando por ahora.")
    print("    Para comparación completa, entrenar MARLISA por separado.")
    
    # 4. SAC Independiente
    print("\n[!] NOTA: SAC requiere entrenamiento. Saltando por ahora.")
    
    # Guardar resultados
    output_path = OUTPUT_DIR / "baselines_evaluation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Resultados guardados en: {output_path}")
    
    # Mostrar resumen
    print("\n" + "="*70)
    print("RESUMEN DE RESULTADOS")
    print("="*70)
    print(f"{'Agente':<25} {'Reward':<15} {'Cost':<10} {'CO2':<10} {'Peak':<10}")
    print("-"*70)
    
    for r in results:
        kpis = r.get("kpis", {})
        print(f"{r['agent']:<25} {r['mean_reward']:<15.2f} "
              f"{kpis.get('cost_total', 1.0):<10.4f} "
              f"{kpis.get('carbon_emissions_total', 1.0):<10.4f} "
              f"{kpis.get('all_time_peak_average', 1.0):<10.4f}")
    
    print("\n[!] Para comparación completa con MADDPG:")
    print("    1. Entrenar MADDPG: python scripts/train_citylearn.py")
    print("    2. Los KPIs se guardan automáticamente en models/citylearn_maddpg/kpis.json")
    print("    3. Comparar con baselines usando los mismos KPIs")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
