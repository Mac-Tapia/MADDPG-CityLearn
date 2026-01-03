# Comparación MADDPG vs Baselines

## Métricas de Recompensa

| Método | Reward Medio | Std | Steps |
|--------|--------------|-----|-------|
| No Control | 883.223 | 0.000 | 8759 |
| Random Agent | 1532.797 | 0.000 | 8759 |
| RBC (Rule-Based) | -6351.091 | 0.000 | 8759 |
| RBC Price-Responsive | -6574.312 | 0.000 | 8759 |
| MADDPG | 8140.920 | 0.000 | 8760 |

## KPIs de CityLearn (Flexibilidad Energética)

| Método | Costo | Emisiones CO2 | Pico | Factor Carga |
|--------|-------|---------------|------|--------------|
| No Control | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Agent | 3.1588 | 3.0649 | 1.4516 | 0.9475 |
| RBC (Rule-Based) | 2.5313 | 2.4783 | 3.5465 | 0.8480 |
| RBC Price-Responsive | 2.5296 | 2.4935 | 2.1794 | 0.8558 |
| MADDPG | 0.9901 | 0.9774 | 0.9696 | 1.4246 |

## Análisis

- **MADDPG** obtuvo reward medio de **8140.920**
- Mejor baseline (Random Agent): **1532.797**
- Mejora de MADDPG sobre mejor baseline: **+6608.123** (+431.1%)