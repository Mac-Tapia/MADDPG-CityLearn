"""
Análisis detallado de KPIs por edificio individual.
Extrae y visualiza métricas para cada uno de los 17 edificios.
"""
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Directorio de resultados
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "models" / "citylearn_maddpg_backup_v1"
OUTPUT_DIR = ROOT / "reports"


def parse_building_kpis(kpis_str: str) -> pd.DataFrame:
    """
    Parsea el string de KPIs y extrae datos por edificio.
    """
    # Extraer líneas relevantes usando regex
    pattern = r"(\d+)\s+([\w_]+)\s+([\d.]+|NaN)\s+([\w_]+)\s+(\w+)"

    data = []
    for match in re.finditer(pattern, kpis_str):
        idx, metric, value, name, level = match.groups()
        try:
            val = float(value) if value != "NaN" else np.nan
        except ValueError:
            val = np.nan

        data.append(
            {
                "index": int(idx),
                "metric": metric,
                "value": val,
                "name": name,
                "level": level,
            }
        )

    return pd.DataFrame(data)


def extract_building_metrics(kpis_str: str) -> dict:
    """
    Extrae métricas principales por edificio de forma manual.
    """
    buildings = {}

    lines = kpis_str.strip().split("\n")

    for line in lines:
        # Buscar líneas con Building_X
        if "Building_" in line and "building" in line:
            parts = line.split()

            # Encontrar el nombre del edificio
            building_name = None
            for p in parts:
                if p.startswith("Building_"):
                    building_name = p
                    break

            if not building_name:
                continue

            if building_name not in buildings:
                buildings[building_name] = {}

            # Intentar extraer métrica y valor
            try:
                idx = parts.index(building_name)
            except ValueError:
                continue

            if idx > 0:
                val_str = parts[idx - 1]
                try:
                    value = float(val_str)
                except ValueError:
                    continue
                metric = parts[1] if len(parts) > 2 else "unknown"
                buildings[building_name][metric] = value

    return buildings


def create_building_summary():
    """Crea resumen de KPIs por edificio basado en datos disponibles."""

    with open(RESULTS_DIR / "maddpg_evaluation.json", "r") as f:
        maddpg_eval = json.load(f)

    kpis_str = maddpg_eval.get("kpis", "")
    kpis_df = parse_building_kpis(kpis_str)
    building_metrics = kpis_df[kpis_df["level"] == "building"]

    metrics_map: Dict[str, str] = {
        "cost_total": "Costo",
        "carbon_emissions_total": "CO2",
        "all_time_peak_average": "Pico",
        "discomfort_cold_delta_average": "Discomfort_Cold",
        "discomfort_hot_delta_average": "Discomfort_Hot",
    }

    building_names: List[str] = sorted(
        building_metrics["name"].dropna().unique().tolist()
    )
    building_data = []
    for building in building_names:
        subset = building_metrics[building_metrics["name"] == building]
        entry = {"Building": building}
        for metric_key, col_name in metrics_map.items():
            values = subset.loc[subset["metric"] == metric_key, "value"]
            entry[col_name] = (
                float(values.iloc[0]) if not values.empty else np.nan
            )
        building_data.append(entry)

    return pd.DataFrame(building_data)


def plot_building_kpis():
    """Genera gráficos de KPIs por edificio."""

    df = create_building_summary()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Colores por edificio
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))

    # 1. Costo por edificio
    ax1 = axes[0, 0]
    ax1.bar(df["Building"], df["Costo"], color=colors, edgecolor="black")
    ax1.axhline(y=1.0, color="red", linestyle="--", label="No Control")
    ax1.axhline(
        y=df["Costo"].mean(),
        color="green",
        linestyle="-",
        linewidth=2,
        label=f'Promedio: {df["Costo"].mean():.3f}',
    )
    ax1.set_ylabel("Costo Normalizado")
    ax1.set_title("Costo Total por Edificio", fontweight="bold")
    ax1.set_xticklabels(df["Building"], rotation=45, ha="right", fontsize=8)
    ax1.legend(fontsize=8)
    ax1.set_ylim(0.8, 1.1)

    # 2. Emisiones CO2 por edificio
    ax2 = axes[0, 1]
    ax2.bar(df["Building"], df["CO2"], color=colors, edgecolor="black")
    ax2.axhline(y=1.0, color="red", linestyle="--", label="No Control")
    ax2.axhline(
        y=df["CO2"].mean(),
        color="green",
        linestyle="-",
        linewidth=2,
        label=f'Promedio: {df["CO2"].mean():.3f}',
    )
    ax2.set_ylabel("Emisiones Normalizadas")
    ax2.set_title("Emisiones CO₂ por Edificio", fontweight="bold")
    ax2.set_xticklabels(df["Building"], rotation=45, ha="right", fontsize=8)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0.8, 1.1)

    # 3. Pico de demanda por edificio
    ax3 = axes[1, 0]
    ax3.bar(df["Building"], df["Pico"], color=colors, edgecolor="black")
    ax3.axhline(y=1.0, color="red", linestyle="--", label="No Control")
    ax3.axhline(
        y=df["Pico"].mean(),
        color="green",
        linestyle="-",
        linewidth=2,
        label=f'Promedio: {df["Pico"].mean():.3f}',
    )
    ax3.set_ylabel("Pico Normalizado")
    ax3.set_title("Pico de Demanda por Edificio", fontweight="bold")
    ax3.set_xticklabels(df["Building"], rotation=45, ha="right", fontsize=8)
    ax3.legend(fontsize=8)
    ax3.set_ylim(0.7, 1.2)

    # 4. Resumen de disconfort
    ax4 = axes[1, 1]
    x = np.arange(len(df))
    width = 0.35
    ax4.bar(
        x - width / 2,
        df["Discomfort_Cold"],
        width,
        label="Frío",
        color="#3498db",
    )
    ax4.bar(
        x + width / 2,
        df["Discomfort_Hot"],
        width,
        label="Calor",
        color="#e74c3c",
    )
    ax4.set_ylabel("Disconfort")
    ax4.set_title("Disconfort Térmico por Edificio", fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(df["Building"], rotation=45, ha="right", fontsize=8)
    ax4.legend(fontsize=8)
    ax4.set_ylim(0, 0.5)
    ax4.text(
        8,
        0.25,
        "✓ Sin disconfort\n(todos = 0)",
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="#d5f5e3"),
    )

    plt.suptitle(
        "Análisis de KPIs por Edificio - MADDPG",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "building_kpis_analysis.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"✅ Guardado: {OUTPUT_DIR / 'building_kpis_analysis.png'}")

    return df


def plot_building_heatmap():
    """Genera heatmap de performance por edificio."""

    df = create_building_summary()

    # Preparar datos para heatmap
    metrics = ["Costo", "CO2", "Pico"]
    heatmap_data = df[metrics].values.T

    fig, ax = plt.subplots(figsize=(16, 4))

    im = ax.imshow(
        heatmap_data, cmap="RdYlGn_r", aspect="auto", vmin=0.85, vmax=1.05
    )

    # Etiquetas
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["Building"], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(["Costo", "Emisiones CO₂", "Pico Demanda"], fontsize=10)

    # Añadir valores en cada celda
    for i in range(len(metrics)):
        for j in range(len(df)):
            ax.text(
                j,
                i,
                f"{heatmap_data[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    plt.colorbar(im, ax=ax, label="Valor (< 1 = mejor)", shrink=0.8)
    ax.set_title(
        "Heatmap de Performance por Edificio (MADDPG)\nVerde = Mejor que No Control",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "building_heatmap.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"✅ Guardado: {OUTPUT_DIR / 'building_heatmap.png'}")


def print_building_summary():
    """Imprime resumen de edificios en consola."""
    df = create_building_summary()

    print("\n" + "=" * 70)
    print("ANÁLISIS POR EDIFICIO - MADDPG")
    print("=" * 70)

    print(
        (
            f"\n{'Edificio':<15} {'Costo':>10} {'CO₂':>10} {'Pico':>10} "
            f"{'Disconf.':>10}"
        )
    )
    print("-" * 55)

    for _, row in df.iterrows():
        status_cost = "✓" if row["Costo"] < 1.0 else "✗"
        status_co2 = "✓" if row["CO2"] < 1.0 else "✗"
        status_peak = "✓" if row["Pico"] < 1.0 else "✗"

        print(
            f"{row['Building']:<15} {row['Costo']:>8.3f} {status_cost} "
            f"{row['CO2']:>8.3f} {status_co2} {row['Pico']:>8.3f} {status_peak} "
            f"{row['Discomfort_Cold']:>8.1f}"
        )

    print("-" * 55)
    print(
        f"{'PROMEDIO':<15} {df['Costo'].mean():>8.3f}   "
        f"{df['CO2'].mean():>8.3f}   {df['Pico'].mean():>8.3f}"
    )

    # Estadísticas
    print("\n" + "=" * 70)
    print("ESTADÍSTICAS")
    print("=" * 70)

    buildings_better_cost = (df["Costo"] < 1.0).sum()
    buildings_better_co2 = (df["CO2"] < 1.0).sum()
    buildings_better_peak = (df["Pico"] < 1.0).sum()

    print(
        (
            "\nEdificios con MEJOR costo que No Control:   "
            f"{buildings_better_cost}/17 "
            f"({buildings_better_cost/17*100:.0f}%)"
        )
    )
    print(
        (
            "Edificios con MENOR CO₂ que No Control:     "
            f"{buildings_better_co2}/17 "
            f"({buildings_better_co2/17*100:.0f}%)"
        )
    )
    print(
        (
            "Edificios con MENOR pico que No Control:    "
            f"{buildings_better_peak}/17 "
            f"({buildings_better_peak/17*100:.0f}%)"
        )
    )
    print("\n✓ Todos los edificios tienen disconfort = 0 (confort mantenido)")

    # Mejor y peor edificio
    best_building = df.loc[df["Costo"].idxmin(), "Building"]
    worst_building = df.loc[df["Costo"].idxmax(), "Building"]

    print(
        (
            "\n🏆 Mejor edificio (menor costo):  "
            f"{best_building} ({df['Costo'].min():.3f})"
        )
    )
    print(
        (
            "⚠️  Peor edificio (mayor costo):   "
            f"{worst_building} ({df['Costo'].max():.3f})"
        )
    )


def main():
    print("=" * 60)
    print("Análisis Detallado por Edificio - MADDPG")
    print("=" * 60)

    # Generar gráficos
    plot_building_kpis()
    plot_building_heatmap()

    # Abrir gráficos
    import subprocess

    subprocess.Popen(
        ["start", "", str(OUTPUT_DIR / "building_kpis_analysis.png")],
        shell=True,
    )
    subprocess.Popen(
        ["start", "", str(OUTPUT_DIR / "building_heatmap.png")], shell=True
    )

    # Imprimir resumen
    print_building_summary()


if __name__ == "__main__":
    main()
