"""
Gráfico de Flexibilidad Energética - Resultados MADDPG para Tesis.
Visualiza los objetivos de control de flexibilidad energética alcanzados.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_energy_flexibility_results():
    """
    Genera gráfico principal de resultados de flexibilidad energética.
    Muestra los 4 objetivos principales de la tesis.
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Datos de resultados
    methods = [
        "No Control\n(Baseline)",
        "Random\nAgent",
        "RBC\nRule-Based",
        "RBC\nPrice-Resp.",
        "MADDPG\n(Propuesto)",
    ]

    # =========================================================================
    # 1. PEAK SHAVING - Reducción de picos de demanda
    # =========================================================================
    ax1 = axes[0, 0]
    peak_values = [1.0, 1.45, 3.55, 2.18, 0.97]
    colors1 = ["#95a5a6", "#e74c3c", "#e74c3c", "#e74c3c", "#27ae60"]

    bars1 = ax1.bar(
        methods, peak_values, color=colors1, edgecolor="black", linewidth=1.5
    )
    ax1.axhline(
        y=1.0, color="#2c3e50", linestyle="--", linewidth=2, label="Baseline"
    )
    ax1.set_ylabel("Pico de Demanda Normalizado", fontsize=11)
    ax1.set_title(
        "PEAK SHAVING\nReducción de Picos de Demanda",
        fontsize=13,
        fontweight="bold",
        color="#27ae60",
    )
    ax1.set_ylim(0, 4)

    # Anotaciones
    for bar, val in zip(bars1, peak_values):
        color = "green" if val < 1.0 else "red" if val > 1.0 else "black"
        ax1.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=color,
        )

    # Flecha de mejora
    ax1.annotate(
        "",
        xy=(4, 0.97),
        xytext=(4, 1.0),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
    )
    ax1.text(4.4, 0.85, "-3%", fontsize=12, color="green", fontweight="bold")

    # =========================================================================
    # 2. COST OPTIMIZATION - Respuesta a precio dinámico
    # =========================================================================
    ax2 = axes[0, 1]
    cost_values = [1.0, 3.16, 2.53, 2.53, 0.99]
    colors2 = ["#95a5a6", "#e74c3c", "#e74c3c", "#e74c3c", "#27ae60"]

    bars2 = ax2.bar(
        methods, cost_values, color=colors2, edgecolor="black", linewidth=1.5
    )
    ax2.axhline(y=1.0, color="#2c3e50", linestyle="--", linewidth=2)
    ax2.set_ylabel("Costo Energético Normalizado", fontsize=11)
    ax2.set_title(
        "COST OPTIMIZATION\nRespuesta a Precio Dinámico",
        fontsize=13,
        fontweight="bold",
        color="#3498db",
    )
    ax2.set_ylim(0, 3.5)

    for bar, val in zip(bars2, cost_values):
        color = "green" if val < 1.0 else "red" if val > 1.0 else "black"
        ax2.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=color,
        )

    ax2.annotate(
        "",
        xy=(4, 0.99),
        xytext=(4, 1.0),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
    )
    ax2.text(4.4, 0.90, "-1%", fontsize=12, color="green", fontweight="bold")

    # =========================================================================
    # 3. CARBON REDUCTION - Reducción de emisiones CO2
    # =========================================================================
    ax3 = axes[1, 0]
    co2_values = [1.0, 3.06, 2.48, 2.49, 0.98]
    colors3 = ["#95a5a6", "#e74c3c", "#e74c3c", "#e74c3c", "#27ae60"]

    bars3 = ax3.bar(
        methods, co2_values, color=colors3, edgecolor="black", linewidth=1.5
    )
    ax3.axhline(y=1.0, color="#2c3e50", linestyle="--", linewidth=2)
    ax3.set_ylabel("Emisiones CO2 Normalizadas", fontsize=11)
    ax3.set_title(
        "CARBON REDUCTION\nReduccion de Emisiones CO2",
        fontsize=13,
        fontweight="bold",
        color="#9b59b6",
    )
    ax3.set_ylim(0, 3.5)

    for bar, val in zip(bars3, co2_values):
        color = "green" if val < 1.0 else "red" if val > 1.0 else "black"
        ax3.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=color,
        )

    ax3.annotate(
        "",
        xy=(4, 0.98),
        xytext=(4, 1.0),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
    )
    ax3.text(4.4, 0.88, "-2.3%", fontsize=12, color="green", fontweight="bold")

    # =========================================================================
    # 4. REWARD TOTAL - Performance global del agente
    # =========================================================================
    ax4 = axes[1, 1]
    reward_values = [883, 1533, -6351, -6574, 8141]
    colors4 = ["#f39c12", "#3498db", "#e74c3c", "#e74c3c", "#27ae60"]

    bars4 = ax4.bar(
        methods, reward_values, color=colors4, edgecolor="black", linewidth=1.5
    )
    ax4.axhline(y=0, color="#2c3e50", linestyle="-", linewidth=1)
    ax4.set_ylabel("Reward Acumulado", fontsize=11)
    ax4.set_title(
        "REWARD TOTAL\nPerformance Global del Agente",
        fontsize=13,
        fontweight="bold",
        color="#e67e22",
    )

    for bar, val in zip(bars4, reward_values):
        offset = 5 if val >= 0 else -15
        va = "bottom" if val >= 0 else "top"
        ax4.annotate(
            f"{val:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # Mejora vs mejor baseline
    ax4.annotate(
        "+431%",
        xy=(4, 8141),
        xytext=(4.3, 6000),
        fontsize=14,
        color="green",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
    )

    # =========================================================================
    # Título general y leyenda
    # =========================================================================
    fig.suptitle(
        "RESULTADOS DE FLEXIBILIDAD ENERGÉTICA\nMADDPG para Control de Comunidades Interactivas con la Red Eléctrica",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    # Leyenda personalizada
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="#27ae60", edgecolor="black", label="MADDPG (Propuesto)"
        ),
        Patch(
            facecolor="#95a5a6",
            edgecolor="black",
            label="No Control (Baseline)",
        ),
        Patch(
            facecolor="#e74c3c", edgecolor="black", label="Peor que Baseline"
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=11,
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "flexibilidad_energetica_resultados.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()
    print(
        f"✅ Guardado: {OUTPUT_DIR / 'flexibilidad_energetica_resultados.png'}"
    )


def plot_flexibility_summary():
    """
    Gráfico resumen de flexibilidad energética tipo dashboard.
    """
    fig = plt.figure(figsize=(16, 10))

    # Grid personalizado
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

    # =========================================================================
    # Sección superior: Métricas principales
    # =========================================================================

    # Métrica 1: Peak Shaving
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(
        0.5,
        0.7,
        "-3.0%",
        fontsize=36,
        ha="center",
        va="center",
        color="#27ae60",
        fontweight="bold",
        transform=ax1.transAxes,
    )
    ax1.text(
        0.5,
        0.25,
        "Peak Shaving",
        fontsize=14,
        ha="center",
        va="center",
        transform=ax1.transAxes,
        fontweight="bold",
    )
    ax1.text(
        0.5,
        0.08,
        "Reducción de picos",
        fontsize=10,
        ha="center",
        va="center",
        transform=ax1.transAxes,
        color="gray",
    )
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.add_patch(
        plt.Rectangle(
            (0.05, 0.02),
            0.9,
            0.96,
            fill=False,
            edgecolor="#27ae60",
            linewidth=3,
        )
    )

    # Métrica 2: Cost Reduction
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(
        0.5,
        0.7,
        "-1.0%",
        fontsize=36,
        ha="center",
        va="center",
        color="#3498db",
        fontweight="bold",
        transform=ax2.transAxes,
    )
    ax2.text(
        0.5,
        0.25,
        "Costo",
        fontsize=14,
        ha="center",
        va="center",
        transform=ax2.transAxes,
        fontweight="bold",
    )
    ax2.text(
        0.5,
        0.08,
        "Ahorro energético",
        fontsize=10,
        ha="center",
        va="center",
        transform=ax2.transAxes,
        color="gray",
    )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.add_patch(
        plt.Rectangle(
            (0.05, 0.02),
            0.9,
            0.96,
            fill=False,
            edgecolor="#3498db",
            linewidth=3,
        )
    )

    # Métrica 3: CO2 Reduction
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(
        0.5,
        0.7,
        "-2.3%",
        fontsize=36,
        ha="center",
        va="center",
        color="#9b59b6",
        fontweight="bold",
        transform=ax3.transAxes,
    )
    ax3.text(
        0.5,
        0.25,
        "Emisiones CO₂",
        fontsize=14,
        ha="center",
        va="center",
        transform=ax3.transAxes,
        fontweight="bold",
    )
    ax3.text(
        0.5,
        0.08,
        "Huella de carbono",
        fontsize=10,
        ha="center",
        va="center",
        transform=ax3.transAxes,
        color="gray",
    )
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    ax3.add_patch(
        plt.Rectangle(
            (0.05, 0.02),
            0.9,
            0.96,
            fill=False,
            edgecolor="#9b59b6",
            linewidth=3,
        )
    )

    # Métrica 4: Reward Improvement
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.text(
        0.5,
        0.7,
        "+431%",
        fontsize=36,
        ha="center",
        va="center",
        color="#e67e22",
        fontweight="bold",
        transform=ax4.transAxes,
    )
    ax4.text(
        0.5,
        0.25,
        "Reward",
        fontsize=14,
        ha="center",
        va="center",
        transform=ax4.transAxes,
        fontweight="bold",
    )
    ax4.text(
        0.5,
        0.08,
        "vs mejor baseline",
        fontsize=10,
        ha="center",
        va="center",
        transform=ax4.transAxes,
        color="gray",
    )
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis("off")
    ax4.add_patch(
        plt.Rectangle(
            (0.05, 0.02),
            0.9,
            0.96,
            fill=False,
            edgecolor="#e67e22",
            linewidth=3,
        )
    )

    # =========================================================================
    # Sección central: Gráfico de barras comparativo
    # =========================================================================
    ax_main = fig.add_subplot(gs[1:, :3])

    metrics = [
        "Pico\nDemanda",
        "Costo\nEnergético",
        "Emisiones\nCO₂",
        "Factor\nCarga",
    ]
    no_control = [1.0, 1.0, 1.0, 1.0]
    maddpg = [0.97, 0.99, 0.98, 1.42]
    random_agent = [1.45, 3.16, 3.06, 0.95]

    x = np.arange(len(metrics))
    width = 0.25

    bars1 = ax_main.bar(
        x - width,
        no_control,
        width,
        label="No Control",
        color="#95a5a6",
        edgecolor="black",
    )
    bars2 = ax_main.bar(
        x, maddpg, width, label="MADDPG", color="#27ae60", edgecolor="black"
    )
    bars3 = ax_main.bar(
        x + width,
        random_agent,
        width,
        label="Random Agent",
        color="#e74c3c",
        edgecolor="black",
    )

    ax_main.axhline(
        y=1.0, color="#2c3e50", linestyle="--", linewidth=2, alpha=0.7
    )
    ax_main.set_ylabel("Valor Normalizado (< 1 = mejor)", fontsize=12)
    ax_main.set_title(
        "Comparación de Métricas de Flexibilidad Energética",
        fontsize=14,
        fontweight="bold",
    )
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(metrics, fontsize=11)
    ax_main.legend(loc="upper left", fontsize=10)
    ax_main.set_ylim(0, 3.5)

    # Valores en barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax_main.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    # =========================================================================
    # Sección derecha: Objetivos de flexibilidad
    # =========================================================================
    ax_obj = fig.add_subplot(gs[1:, 3])

    objectives = [
        ("Peak Shaving", "LOGRADO"),
        ("Cost Reduction", "LOGRADO"),
        ("CO2 Reduction", "LOGRADO"),
        ("Load Shifting", "ACTIVO"),
        ("Self-Consumption", "OPTIMIZADO"),
        ("EV Charging", "COORDINADO"),
        ("Thermal Comfort", "MANTENIDO"),
    ]

    for i, (obj, status) in enumerate(objectives):
        y_pos = 0.9 - i * 0.12
        ax_obj.text(
            0.05,
            y_pos,
            obj,
            fontsize=11,
            transform=ax_obj.transAxes,
            va="center",
        )
        color = "#27ae60"
        ax_obj.text(
            0.70,
            y_pos,
            status,
            fontsize=9,
            transform=ax_obj.transAxes,
            va="center",
            color=color,
            fontweight="bold",
        )

    ax_obj.set_xlim(0, 1)
    ax_obj.set_ylim(0, 1)
    ax_obj.axis("off")
    ax_obj.set_title(
        "Objetivos de\nFlexibilidad", fontsize=12, fontweight="bold"
    )
    ax_obj.add_patch(
        plt.Rectangle(
            (0.02, 0.02),
            0.96,
            0.96,
            fill=False,
            edgecolor="#34495e",
            linewidth=2,
            linestyle="--",
        )
    )

    # =========================================================================
    # Título principal
    # =========================================================================
    fig.suptitle(
        "CONTROL DE FLEXIBILIDAD ENERGÉTICA EN COMUNIDADES INTERACTIVAS\n"
        "Multi-Agent Deep Reinforcement Learning (MADDPG) - CityLearn",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(
        OUTPUT_DIR / "flexibilidad_energetica_dashboard.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()
    print(
        f"✅ Guardado: {OUTPUT_DIR / 'flexibilidad_energetica_dashboard.png'}"
    )


def main():
    print("=" * 60)
    print("Generando Gráficos de Flexibilidad Energética")
    print("=" * 60)

    plot_energy_flexibility_results()
    plot_flexibility_summary()

    # Abrir gráficos
    import subprocess

    subprocess.Popen(
        [
            "start",
            "",
            str(OUTPUT_DIR / "flexibilidad_energetica_resultados.png"),
        ],
        shell=True,
    )
    subprocess.Popen(
        [
            "start",
            "",
            str(OUTPUT_DIR / "flexibilidad_energetica_dashboard.png"),
        ],
        shell=True,
    )

    print("\n✅ Gráficos generados y abiertos")


if __name__ == "__main__":
    main()
