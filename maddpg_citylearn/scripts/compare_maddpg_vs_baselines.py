"""
Script para comparar MADDPG entrenado vs baselines.

Genera tabla comparativa de KPIs y métricas para la tesis.

Ejecutar después del entrenamiento:
    PYTHONPATH=src python scripts/compare_maddpg_vs_baselines.py
"""

import json
import os
from typing import Dict, Any, Optional

import numpy as np

from maddpg_tesis.core.config import load_config
from maddpg_tesis.core.logging import get_logger, setup_logging
from maddpg_tesis.maddpg import MADDPG
from maddpg_tesis.envs import CityLearnMultiAgentEnv


setup_logging()
logger = get_logger("compare_results")


def evaluate_maddpg(
    checkpoint_path: str,
    schema: str,
    n_episodes: int = 1,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Evalúa el modelo MADDPG entrenado."""

    env = CityLearnMultiAgentEnv(schema=schema, central_agent=False)

    cfg = load_config()
    maddpg = MADDPG(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        cfg=cfg.maddpg,
    )

    # Cargar checkpoint
    maddpg.load(checkpoint_path)
    logger.info("Modelo cargado desde: %s", checkpoint_path)

    all_rewards = []
    all_steps = []

    for ep in range(n_episodes):
        obs = env.reset()
        episode_rewards = np.zeros(env.n_agents, dtype=np.float32)
        steps = 0
        done = False

        while not done:
            actions = maddpg.select_actions(obs, noise=False)
            obs, rewards, done, info = env.step(actions)
            episode_rewards += rewards
            steps += 1

            if max_steps and steps >= max_steps:
                break

        all_rewards.append(episode_rewards.mean())
        all_steps.append(steps)
        logger.info(
            "MADDPG | Ep %d | steps=%d | reward_mean=%.3f",
            ep + 1,
            steps,
            episode_rewards.mean(),
        )

    # KPIs
    try:
        kpis = env.evaluate()
    except Exception as e:
        logger.warning("No se pudieron obtener KPIs: %s", e)
        kpis = {}

    env.close()

    return {
        "method": "MADDPG",
        "checkpoint": checkpoint_path,
        "episodes": n_episodes,
        "total_steps": sum(all_steps),
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "kpis": kpis,
    }


def parse_kpis_from_string(kpi_str: str) -> Dict[str, float]:
    """Extrae KPIs de nivel District desde string del DataFrame."""
    kpis = {}
    if not isinstance(kpi_str, str):
        return kpis

    # Parsear líneas del DataFrame string
    lines = kpi_str.split("\n")
    for line in lines:
        # Buscar líneas que contengan 'District' y valores numéricos
        if "District" in line:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    # El valor suele estar después del nombre de la métrica
                    # Formato: idx  metric_name  value  District  district
                    for i, part in enumerate(parts):
                        if part == "District":
                            # Buscar el valor numérico antes de 'District'
                            for j in range(i - 1, -1, -1):
                                try:
                                    val = float(parts[j])
                                    # Encontrar nombre de métrica
                                    metric_parts = parts[1:j]
                                    if metric_parts:
                                        metric_name = "_".join(metric_parts)
                                        kpis[metric_name] = val
                                    break
                                except ValueError:
                                    continue
                            break
                except Exception:
                    continue
    return kpis


def extract_district_kpis(kpis: Any) -> Dict[str, float]:
    """Extrae KPIs principales del distrito."""
    result = {}

    if isinstance(kpis, str):
        return parse_kpis_from_string(kpis)
    elif hasattr(kpis, "to_dict"):  # DataFrame
        df = kpis
        district_kpis = df[df["level"] == "district"]
        for _, row in district_kpis.iterrows():
            result[row["cost_function"]] = row["value"]
    elif isinstance(kpis, dict):
        return kpis
    elif isinstance(kpis, list):
        # Lista de dicts (to_dict(orient='records'))
        for row in kpis:
            if row.get("level") == "district":
                result[row["cost_function"]] = row["value"]

    return result


def generate_comparison_table(
    maddpg_results: Dict[str, Any],
    baselines_path: str = "models/citylearn_maddpg/baselines_comparison.json",
) -> str:
    """Genera tabla Markdown para la tesis."""

    with open(baselines_path, "r", encoding="utf-8") as f:
        baselines = json.load(f)

    # Combinar resultados
    all_results = baselines + [maddpg_results]

    # Generar tabla
    lines = [
        "# Comparación MADDPG vs Baselines",
        "",
        "## Métricas de Recompensa",
        "",
        "| Método | Reward Medio | Std | Steps |",
        "|--------|--------------|-----|-------|",
    ]

    for r in all_results:
        name = r.get("method", r.get("baseline", "Unknown"))
        lines.append(
            f"| {name} | {r['mean_reward']:.3f} | {r['std_reward']:.3f} | {r['total_steps']} |"
        )

    # KPIs si están disponibles
    lines.extend(
        [
            "",
            "## KPIs de CityLearn (Flexibilidad Energética)",
            "",
        ]
    )

    header = "| Método | Costo | Emisiones CO2 | Pico | Factor Carga |"
    separator = "|--------|-------|---------------|------|--------------|"
    lines.append(header)
    lines.append(separator)

    for r in all_results:
        name = r.get("method", r.get("baseline", "Unknown"))
        kpis = extract_district_kpis(r.get("kpis", {}))

        cost = kpis.get("cost_total", "N/A")
        co2 = kpis.get("carbon_emissions_total", "N/A")
        peak = kpis.get("all_time_peak_average", "N/A")
        load_factor = kpis.get("daily_one_minus_load_factor_average", "N/A")

        def fmt(v):
            if isinstance(v, (int, float)) and not np.isnan(v):
                return f"{v:.4f}"
            return str(v)

        lines.append(
            f"| {name} | {fmt(cost)} | {fmt(co2)} | {fmt(peak)} | {fmt(load_factor)} |"
        )

    # Análisis
    lines.extend(
        [
            "",
            "## Análisis",
            "",
            f"- **MADDPG** obtuvo reward medio de **{maddpg_results['mean_reward']:.3f}**",
        ]
    )

    # Comparar con mejor baseline
    best_baseline = max(baselines, key=lambda x: x["mean_reward"])
    improvement = maddpg_results["mean_reward"] - best_baseline["mean_reward"]
    pct = (
        (improvement / abs(best_baseline["mean_reward"])) * 100
        if best_baseline["mean_reward"] != 0
        else 0
    )

    lines.append(
        f"- Mejor baseline ({best_baseline['baseline']}): **{best_baseline['mean_reward']:.3f}**"
    )
    lines.append(
        f"- Mejora de MADDPG sobre mejor baseline: **{improvement:+.3f}** ({pct:+.1f}%)"
    )

    return "\n".join(lines)


def main():
    cfg = load_config()

    # Evaluar MADDPG
    checkpoint = os.path.join(cfg.training.save_dir, "maddpg.pt")

    if not os.path.exists(checkpoint):
        # Intentar con val_best o last
        for alt in ["maddpg_val_best.pt", "maddpg_last.pt"]:
            alt_path = os.path.join(cfg.training.save_dir, alt)
            if os.path.exists(alt_path):
                checkpoint = alt_path
                break

    if not os.path.exists(checkpoint):
        logger.error(
            "No se encontró checkpoint de MADDPG en %s", cfg.training.save_dir
        )
        logger.info("Ejecuta primero el entrenamiento o evaluate_baselines.py")
        return

    logger.info("=" * 60)
    logger.info("COMPARACIÓN MADDPG VS BASELINES")
    logger.info("=" * 60)

    maddpg_results = evaluate_maddpg(
        checkpoint_path=checkpoint,
        schema=cfg.env.schema,
        n_episodes=1,
    )

    # Guardar resultados MADDPG
    maddpg_path = os.path.join(cfg.training.save_dir, "maddpg_evaluation.json")
    with open(maddpg_path, "w", encoding="utf-8") as f:
        json.dump(maddpg_results, f, indent=2, default=str)

    # Generar tabla comparativa
    baselines_path = os.path.join(
        cfg.training.save_dir, "baselines_comparison.json"
    )

    if os.path.exists(baselines_path):
        table = generate_comparison_table(maddpg_results, baselines_path)

        # Guardar tabla
        table_path = os.path.join(
            cfg.training.save_dir, "comparison_results.md"
        )
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(table)

        logger.info("\n%s", table)
        logger.info("\nTabla guardada en: %s", table_path)
    else:
        logger.warning("No se encontró baselines_comparison.json")
        logger.info("Ejecuta primero: python scripts/evaluate_baselines.py")

    logger.info("\nResultados MADDPG guardados en: %s", maddpg_path)


if __name__ == "__main__":
    main()
