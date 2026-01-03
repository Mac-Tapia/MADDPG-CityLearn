"""
Script para generar gráficas mejoradas de KPIs y resultados.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Configuración
REPORTS_DIR = "reports/continue_training"

# Cargar datos
with open(os.path.join(REPORTS_DIR, 'kpis.json'), 'r') as f:
    kpis = json.load(f)

with open(os.path.join(REPORTS_DIR, 'training_history.json'), 'r') as f:
    history = json.load(f)

# Filtrar KPIs de distrito (más importantes)
district_kpis = [k for k in kpis if k.get('level') == 'district']

print("=" * 60)
print("📊 GENERANDO GRÁFICAS MEJORADAS DE KPIs")
print("=" * 60)
print(f"Total KPIs: {len(kpis)}")
print(f"KPIs distrito: {len(district_kpis)}")

# 1. GRÁFICA DE KPIs PRINCIPALES DEL DISTRITO
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sistema Multi-Agente MADDPG para Optimización de Flexibilidad Energética\n'
             'KPIs de CityLearn', fontsize=14, fontweight='bold')

# KPIs principales a mostrar
main_kpis = {
    'cost_total': 'Costo Total',
    'carbon_emissions_total': 'Emisiones CO₂',
    'daily_peak_average': 'Pico Diario Promedio',
    'electricity_consumption_total': 'Consumo Eléctrico'
}

# Extraer valores
kpi_values = {}
for k in district_kpis:
    name = k['cost_function']
    if name in main_kpis:
        val = k['value']
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            kpi_values[name] = val

print("\nKPIs principales:")
for k, v in kpi_values.items():
    print(f"  {k}: {v:.4f}")

# Subplot 1: Barras de KPIs principales vs baseline (1.0)
ax1 = axes[0, 0]
names = [main_kpis[k] for k in kpi_values.keys()]
values = list(kpi_values.values())
colors = ['green' if v < 1.0 else 'orange' if v == 1.0 else 'red' for v in values]

bars = ax1.barh(names, values, color=colors, edgecolor='black', alpha=0.8)
ax1.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Baseline (1.0)')
ax1.set_xlabel('Valor (1.0 = Baseline)', fontsize=12)
ax1.set_title('KPIs vs Baseline Sin Control', fontsize=14)
ax1.legend()

# Añadir valores en las barras
for bar, val in zip(bars, values):
    width = bar.get_width()
    pct = (1 - val) * 100
    label = f'{val:.3f} ({pct:+.1f}%)'
    ax1.text(width + 0.02, bar.get_y() + bar.get_height() / 2, label,
             ha='left', va='center', fontsize=10, fontweight='bold')

ax1.set_xlim(0, max(values) * 1.3)
ax1.grid(True, alpha=0.3, axis='x')

# Subplot 2: KPIs adicionales de flexibilidad
ax2 = axes[0, 1]
flex_kpis = {
    'ramping_average': 'Ramping Promedio',
    'daily_one_minus_load_factor_average': '1-Load Factor',
    'all_time_peak_average': 'Pico Máximo',
    'monthly_one_minus_load_factor_average': '1-Load Factor Mensual'
}

flex_values = {}
for k in district_kpis:
    name = k['cost_function']
    if name in flex_kpis:
        val = k['value']
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            flex_values[name] = val

if flex_values:
    names2 = [flex_kpis[k] for k in flex_values.keys()]
    values2 = list(flex_values.values())
    colors2 = ['green' if v < 1.0 else 'orange' if v == 1.0 else 'coral' for v in values2]

    bars2 = ax2.barh(names2, values2, color=colors2, edgecolor='black', alpha=0.8)
    ax2.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Baseline')
    ax2.set_xlabel('Valor (1.0 = Baseline)', fontsize=12)
    ax2.set_title('KPIs de Flexibilidad Energética', fontsize=14)
    ax2.legend()

    for bar, val in zip(bars2, values2):
        width = bar.get_width()
        pct = (1 - val) * 100
        ax2.text(width + 0.02, bar.get_y() + bar.get_height() / 2, f'{val:.3f}',
                 ha='left', va='center', fontsize=10)
    ax2.set_xlim(0, max(values2) * 1.3)
    ax2.grid(True, alpha=0.3, axis='x')

# Subplot 3: Comparación MADDPG vs Baselines
ax3 = axes[1, 0]
baselines = history.get('baselines', {})
if baselines:
    methods = list(baselines.keys())
    baseline_rewards = [baselines[m].get('mean_reward', 0) for m in methods]
    maddpg_reward = np.mean(history['mean_rewards'])

    all_methods = methods + ['MADDPG']
    all_rewards = baseline_rewards + [maddpg_reward]

    colors3 = ['#e74c3c' if r < 0 else '#95a5a6' for r in baseline_rewards] + ['#27ae60']

    bars3 = ax3.bar(all_methods, all_rewards, color=colors3, edgecolor='black', alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel('Reward Medio', fontsize=12)
    ax3.set_title('Comparación: MADDPG vs Baselines', fontsize=14)
    ax3.tick_params(axis='x', rotation=20)

    # Valores en barras
    for bar, val in zip(bars3, all_rewards):
        height = bar.get_height()
        y_pos = height + 200 if height >= 0 else height - 500
        ax3.text(bar.get_x() + bar.get_width() / 2, y_pos, f'{val:,.0f}',
                 ha='center', va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')

    ax3.grid(True, alpha=0.3, axis='y')

# Subplot 4: Progreso del entrenamiento
ax4 = axes[1, 1]
episodes = history['episodes']
rewards = history['mean_rewards']

ax4.plot(episodes, rewards, 'b-o', linewidth=2, markersize=10, label='MADDPG')
ax4.fill_between(episodes, rewards, alpha=0.3)
ax4.axhline(y=np.mean(rewards), color='r', linestyle='--', linewidth=2,
            label=f'Media: {np.mean(rewards):,.0f}')

# Añadir baselines como líneas de referencia
if baselines:
    colors_bl = ['gray', 'orange', 'purple', 'brown']
    for i, (name, data) in enumerate(baselines.items()):
        bl_reward = data.get('mean_reward', 0)
        if bl_reward > -2000:  # Solo mostrar baselines razonables
            ax4.axhline(y=bl_reward, color=colors_bl[i % len(colors_bl)],
                        linestyle=':', alpha=0.7, label=f'{name}: {bl_reward:,.0f}')

ax4.set_xlabel('Episodio', fontsize=12)
ax4.set_ylabel('Reward Medio', fontsize=12)
ax4.set_title('Progreso del Entrenamiento (5 Episodios)', fontsize=14)
ax4.legend(loc='lower right', fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'kpis_completos.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ Gráfica guardada: {os.path.join(REPORTS_DIR, 'kpis_completos.png')}")

# 2. GRÁFICA DETALLADA DE KPIs POR EDIFICIO
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Sistema Multi-Agente MADDPG - Comunidades Interactivas de Redes Eléctricas\n'
              'KPIs por Edificio', fontsize=14, fontweight='bold')

# Filtrar KPIs por edificio
building_kpis = [k for k in kpis if k.get('level') == 'building']

# Organizar por edificio
buildings = {}
for k in building_kpis:
    bname = k['name']
    if bname not in buildings:
        buildings[bname] = {}
    buildings[bname][k['cost_function']] = k['value']

building_names = sorted(buildings.keys(), key=lambda x: int(x.split('_')[1]))
print(f"\nEdificios: {len(building_names)}")

# Subplot 1: Costo por edificio
ax2_1 = axes2[0, 0]
cost_values = []
for b in building_names:
    val = buildings[b].get('cost_total', np.nan)
    cost_values.append(val if val is not None else np.nan)

valid_costs = [v for v in cost_values if not np.isnan(v)]
if valid_costs:
    x_pos = range(len(building_names))
    colors_cost = ['green' if v < 1.0 else 'red' for v in cost_values]
    bars_c = ax2_1.bar(x_pos, cost_values, color=colors_cost, edgecolor='black', alpha=0.8)
    ax2_1.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Baseline')
    ax2_1.set_xlabel('Edificio', fontsize=12)
    ax2_1.set_ylabel('Costo Total (ratio)', fontsize=12)
    ax2_1.set_title('Costo Total por Edificio', fontsize=14)
    ax2_1.set_xticks(x_pos)
    ax2_1.set_xticklabels([f'B{i + 1}' for i in range(len(building_names))], rotation=45)
    ax2_1.legend()
    ax2_1.grid(True, alpha=0.3, axis='y')

# Subplot 2: Emisiones CO2 por edificio
ax2_2 = axes2[0, 1]
co2_values = []
for b in building_names:
    val = buildings[b].get('carbon_emissions_total', np.nan)
    co2_values.append(val if val is not None else np.nan)

valid_co2 = [v for v in co2_values if not np.isnan(v)]
if valid_co2:
    colors_co2 = ['green' if v < 1.0 else 'red' for v in co2_values]
    bars_co2 = ax2_2.bar(x_pos, co2_values, color=colors_co2, edgecolor='black', alpha=0.8)
    ax2_2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Baseline')
    ax2_2.set_xlabel('Edificio', fontsize=12)
    ax2_2.set_ylabel('Emisiones CO₂ (ratio)', fontsize=12)
    ax2_2.set_title('Emisiones CO₂ por Edificio', fontsize=14)
    ax2_2.set_xticks(x_pos)
    ax2_2.set_xticklabels([f'B{i + 1}' for i in range(len(building_names))], rotation=45)
    ax2_2.legend()
    ax2_2.grid(True, alpha=0.3, axis='y')

# Subplot 3: Peak diario por edificio
ax2_3 = axes2[1, 0]
peak_values = []
for b in building_names:
    val = buildings[b].get('daily_peak_average', np.nan)
    peak_values.append(val if val is not None else np.nan)

valid_peak = [v for v in peak_values if not np.isnan(v)]
if valid_peak:
    colors_peak = ['green' if v < 1.0 else 'orange' for v in peak_values]
    bars_peak = ax2_3.bar(x_pos, peak_values, color=colors_peak, edgecolor='black', alpha=0.8)
    ax2_3.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Baseline')
    ax2_3.set_xlabel('Edificio', fontsize=12)
    ax2_3.set_ylabel('Pico Diario (ratio)', fontsize=12)
    ax2_3.set_title('Peak Shaving por Edificio', fontsize=14)
    ax2_3.set_xticks(x_pos)
    ax2_3.set_xticklabels([f'B{i + 1}' for i in range(len(building_names))], rotation=45)
    ax2_3.legend()
    ax2_3.grid(True, alpha=0.3, axis='y')

# Subplot 4: Resumen de mejoras (barras agrupadas)
ax2_4 = axes2[1, 1]
kpi_names = ['Costo', 'CO₂', 'Peak', 'Consumo']
district_vals = [
    kpi_values.get('cost_total', 1.0),
    kpi_values.get('carbon_emissions_total', 1.0),
    kpi_values.get('daily_peak_average', 1.0),
    kpi_values.get('electricity_consumption_total', 1.0)
]
mejoras = [(1 - v) * 100 for v in district_vals]

colors_mejora = ['green' if m > 0 else 'red' for m in mejoras]
bars_mejora = ax2_4.bar(kpi_names, mejoras, color=colors_mejora, edgecolor='black', alpha=0.8)
ax2_4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2_4.set_ylabel('Mejora vs Baseline (%)', fontsize=12)
ax2_4.set_title('Resumen de Mejoras (Nivel Distrito)', fontsize=14)

for bar, val in zip(bars_mejora, mejoras):
    height = bar.get_height()
    ax2_4.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{val:+.1f}%',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

ax2_4.grid(True, alpha=0.3, axis='y')
ax2_4.set_ylim(min(mejoras) - 5, max(mejoras) + 10)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'kpis_por_edificio.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Gráfica guardada: {os.path.join(REPORTS_DIR, 'kpis_por_edificio.png')}")

# 3. TABLA RESUMEN EN CONSOLA
print("\n" + "=" * 70)
print("📊 RESUMEN DE KPIs - NIVEL DISTRITO")
print("=" * 70)
print(f"{'KPI':<40} {'Valor':>10} {'vs Baseline':>15}")
print("-" * 70)
for kpi_name, display_name in main_kpis.items():
    if kpi_name in kpi_values:
        val = kpi_values[kpi_name]
        pct = (1 - val) * 100
        status = "✅" if pct > 0 else "⚠️"
        print(f"{status} {display_name:<37} {val:>10.4f} {pct:>+13.1f}%")
print("-" * 70)

print("\n🎉 Gráficas generadas correctamente!")
print("   - kpis_completos.png")
print("   - kpis_por_edificio.png")
