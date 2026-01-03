# Reporte de Entrenamiento MADDPG - CityLearn

## Resumen Ejecutivo

| Métrica | MADDPG | Mejor Baseline | Mejora |
|---------|--------|----------------|--------|
| Reward Medio | 8,140.92 | 1,532.80 (Random) | **+431%** |
| Costo Total | 0.990 | 1.000 (No Control) | **-1.0%** |
| Emisiones CO₂ | 0.977 | 1.000 (No Control) | **-2.3%** |
| Pico Demanda | 0.970 | 1.000 (No Control) | **-3.0%** |

## Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| γ (gamma) | 0.95 |
| τ (tau) | 0.005 |
| Actor LR | 0.0003 |
| Critic LR | 0.001 |
| Hidden dim | 256 |
| Batch size | 256 |
| Buffer size | 100,000 |
| Episodios | 10 |
| Steps/episodio | 8,760 |

## Pesos de Recompensa

| Objetivo | Peso |
|----------|------|
| Costo energético | 1.0 |
| Peak shaving | 0.5 |
| Emisiones CO₂ | 0.3 |
| Confort térmico | 0.2 |

## Gráficos Generados

1. `reward_comparison.png` - Comparación de rewards entre métodos
2. `kpi_comparison.png` - KPIs de flexibilidad energética
3. `maddpg_radar.png` - Perfil radar de MADDPG
4. `maddpg_improvements.png` - Mejoras porcentuales

## Conclusiones

1. **MADDPG supera ampliamente todos los baselines** con +431% de mejora en reward
2. **Reduce costos, emisiones y picos** simultáneamente
3. **Los RBCs tienen performance negativo** - no son efectivos para este problema
4. **CTDE funciona**: los 17 agentes aprenden coordinación efectiva
