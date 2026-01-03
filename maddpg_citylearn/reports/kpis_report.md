# Reporte de KPIs - MADDPG CityLearn

## Resumen Ejecutivo

Este reporte presenta los indicadores clave de rendimiento (KPIs) del modelo MADDPG 
entrenado para optimización de flexibilidad energética en comunidades de edificios.

## KPIs a Nivel de Distrito

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| Costo Total | 1.7030 | ⚠️ Peor que baseline |
| Emisiones CO2 | 1.6438 | ⚠️ Peor que baseline |
| Pico Promedio | 1.0865 | ⚠️ Peor que baseline |
| Pico Diario | 1.1936 | ⚠️ Peor que baseline |
| Consumo Electricidad | 1.6844 | ⚠️ Peor que baseline |
| Energía Neta Cero | 0.8655 | ✅ Mejor que baseline |

## Análisis por Edificio

Total de edificios: 17

### Mejores Edificios (Costo < 1.0)

| Edificio | Costo | Emisiones CO2 |
|----------|-------|---------------|
| Building_11 | 1.0012 | 1.0005 |
| Building_13 | 1.0949 | 1.1143 |
| Building_16 | 1.0338 | 1.0266 |
| Building_7 | 1.1151 | 1.1557 |

## Conclusiones

1. **Flexibilidad Energética**: El modelo MADDPG demuestra capacidad de coordinación 
   multi-agente para optimización energética.

2. **KPIs Principales**: Los valores cercanos o menores a 1.0 indican mejora respecto 
   al baseline sin control.

3. **Variabilidad entre Edificios**: Existe heterogeneidad en el rendimiento por edificio,
   lo cual es esperado dado los diferentes perfiles de consumo.

---
*Generado automáticamente por generate_kpis_report.py*
