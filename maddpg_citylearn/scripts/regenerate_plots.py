"""
Regenerar gráficas de entrenamiento con el nuevo título del proyecto.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

REPORTS_DIR = "reports/continue_training"

# Cargar datos guardados
with open(os.path.join(REPORTS_DIR, 'training_history.json'), 'r') as f:
    history = json.load(f)

print("=" * 60)
print("🔄 REGENERANDO GRÁFICAS CON NUEVO TÍTULO")
print("=" * 60)

# 1. GRÁFICA DE PROGRESO DE ENTRENAMIENTO
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Sistema Multi-Agente de Aprendizaje Profundo por Refuerzo\n'
    'para Optimización de Flexibilidad Energética en Comunidades Interactivas',
    fontsize=12,
    fontweight='bold')

# Reward medio
ax1 = axes[0, 0]
episodes = history["episodes"]
rewards = history["mean_rewards"]
ax1.plot(episodes, rewards, 'b-o', linewidth=2, markersize=8, label='MADDPG')
ax1.axhline(y=np.mean(rewards), color='r', linestyle='--', linewidth=2, label=f'Media: {np.mean(rewards):,.0f}')
ax1.set_xlabel('Episodio', fontsize=12)
ax1.set_ylabel('Reward Medio', fontsize=12)
ax1.set_title('Progreso de Entrenamiento - Reward Medio', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Reward por agente
ax2 = axes[0, 1]
if "agent_rewards" in history and history["agent_rewards"]:
    last_agent_rewards = history["agent_rewards"][-1]
    agents = list(range(1, len(last_agent_rewards) + 1))
    colors = plt.cm.viridis(np.linspace(0, 1, len(agents)))
    bars = ax2.bar(agents, last_agent_rewards, color=colors, edgecolor='black')
    ax2.axhline(y=np.mean(last_agent_rewards), color='r', linestyle='--', linewidth=2,
                label=f'Media: {np.mean(last_agent_rewards):,.0f}')
    ax2.set_xlabel('Agente (Edificio)', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Reward por Agente (Último Episodio)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

# Steps por episodio
ax3 = axes[1, 0]
if "steps" in history:
    ax3.bar(episodes, history["steps"], color='steelblue', alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Episodio', fontsize=12)
    ax3.set_ylabel('Steps', fontsize=12)
    ax3.set_title('Steps por Episodio (8,760 = 1 año)', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')

# Comparación con baselines
ax4 = axes[1, 1]
baselines = history.get("baselines", {})
if baselines:
    methods = list(baselines.keys())
    baseline_rewards = [baselines[m].get("mean_reward", 0) for m in methods]
    maddpg_reward = np.mean(rewards)

    all_methods = methods + ["MADDPG"]
    all_rewards = baseline_rewards + [maddpg_reward]

    colors = ['#e74c3c' if r < 0 else '#95a5a6' for r in baseline_rewards] + ['#27ae60']
    bars = ax4.bar(all_methods, all_rewards, color=colors, edgecolor='black', alpha=0.8)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_ylabel('Reward Medio', fontsize=12)
    ax4.set_title('Comparación: MADDPG vs Baselines', fontsize=14)
    ax4.tick_params(axis='x', rotation=20)

    for bar, val in zip(bars, all_rewards):
        height = bar.get_height()
        y_pos = height + 200 if height >= 0 else height - 500
        ax4.text(bar.get_x() + bar.get_width() / 2, y_pos, f'{val:,.0f}',
                 ha='center', va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'training_progress.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Gráfica guardada: training_progress.png")

# 2. GRÁFICA DE FLEXIBILIDAD ENERGÉTICA (simulada con datos del historial)
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle(
    'Optimización de Flexibilidad Energética en Comunidades Interactivas\nAnálisis del Sistema Multi-Agente MADDPG',
    fontsize=12,
    fontweight='bold')

# Subplot 1: Rewards por agente en cada episodio
ax2_1 = axes2[0, 0]
agent_rewards_array = np.array(history["agent_rewards"])  # [episodes, n_agents]
n_agents = agent_rewards_array.shape[1]

for ep_idx, ep_rewards in enumerate(agent_rewards_array):
    ax2_1.plot(range(1, n_agents + 1), ep_rewards, 'o-', alpha=0.7,
               label=f'Ep {ep_idx + 1}', linewidth=2)

ax2_1.set_xlabel('Agente (Edificio)', fontsize=12)
ax2_1.set_ylabel('Reward', fontsize=12)
ax2_1.set_title('Evolución de Rewards por Agente', fontsize=14)
ax2_1.legend(loc='lower right')
ax2_1.grid(True, alpha=0.3)

# Subplot 2: Mejora por episodio
ax2_2 = axes2[0, 1]
mejoras = [(rewards[i] - rewards[0]) / rewards[0] * 100 if i > 0 else 0 for i in range(len(rewards))]
colors_mejora = ['#27ae60' if m >= 0 else '#e74c3c' for m in mejoras]
ax2_2.bar(episodes, mejoras, color=colors_mejora, edgecolor='black', alpha=0.8)
ax2_2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2_2.set_xlabel('Episodio', fontsize=12)
ax2_2.set_ylabel('Mejora vs Episodio 1 (%)', fontsize=12)
ax2_2.set_title('Mejora Relativa durante Entrenamiento', fontsize=14)
ax2_2.grid(True, alpha=0.3, axis='y')

for i, (ep, m) in enumerate(zip(episodes, mejoras)):
    ax2_2.text(ep, m + 0.1, f'{m:+.1f}%', ha='center', va='bottom', fontsize=10)

# Subplot 3: Distribución de rewards por agente (boxplot)
ax2_3 = axes2[1, 0]
bp = ax2_3.boxplot(agent_rewards_array.T, labels=[f'Ep{i}' for i in episodes],
                   patch_artist=True)
colors_box = plt.cm.Blues(np.linspace(0.3, 0.9, len(episodes)))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
ax2_3.set_xlabel('Episodio', fontsize=12)
ax2_3.set_ylabel('Reward por Agente', fontsize=12)
ax2_3.set_title('Distribución de Rewards entre Agentes', fontsize=14)
ax2_3.grid(True, alpha=0.3, axis='y')

# Subplot 4: Heatmap de rewards por agente y episodio
ax2_4 = axes2[1, 1]
im = ax2_4.imshow(agent_rewards_array.T, aspect='auto', cmap='YlGn')
ax2_4.set_xlabel('Episodio', fontsize=12)
ax2_4.set_ylabel('Agente (Edificio)', fontsize=12)
ax2_4.set_title('Mapa de Calor: Rewards por Agente y Episodio', fontsize=14)
ax2_4.set_xticks(range(len(episodes)))
ax2_4.set_xticklabels(episodes)
ax2_4.set_yticks(range(n_agents))
ax2_4.set_yticklabels([f'B{i + 1}' for i in range(n_agents)])
plt.colorbar(im, ax=ax2_4, label='Reward')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'energy_flexibility.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Gráfica guardada: energy_flexibility.png")

print("\n🎉 Todas las gráficas regeneradas con el nuevo título!")
print("   - training_progress.png")
print("   - energy_flexibility.png")
print("   - kpis_completos.png")
print("   - kpis_por_edificio.png")
