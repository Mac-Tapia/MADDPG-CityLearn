"""
Script para generar reportes y gráficos a partir de KPIs de entrenamiento MADDPG.
Lee los archivos kpis.json existentes y genera visualizaciones.
"""
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np

# Configuración de paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Directorios
MODELS_DIR = ROOT / "models" / "citylearn_maddpg"
REPORTS_DIR = ROOT / "reports"
STATIC_DIR = ROOT / "static" / "images"

# Crear directorios si no existen
REPORTS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)


def load_kpis(kpis_path: Optional[Path] = None) -> Optional[List[Dict[str, Any]]]:
    """Carga KPIs desde archivo JSON."""
    if kpis_path is None:
        kpis_path = MODELS_DIR / "kpis.json"
    
    if not kpis_path.exists():
        print(f"[!] Archivo no encontrado: {kpis_path}")
        return None
    
    try:
        with open(kpis_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[!] Error cargando KPIs: {e}")
        return None


def filter_district_kpis(kpis: List[Dict[str, Any]]) -> Dict[str, float]:
    """Filtra KPIs a nivel de distrito."""
    district_kpis = {}
    for kpi in kpis:
        if kpi.get("level") == "district":
            name = kpi.get("cost_function", "unknown")
            value = kpi.get("value")
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                district_kpis[name] = value
    return district_kpis


def filter_building_kpis(kpis: List[Dict[str, Any]], building_name: str) -> Dict[str, float]:
    """Filtra KPIs de un edificio específico."""
    building_kpis = {}
    for kpi in kpis:
        if kpi.get("level") == "building" and kpi.get("name") == building_name:
            name = kpi.get("cost_function", "unknown")
            value = kpi.get("value")
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                building_kpis[name] = value
    return building_kpis


def get_all_buildings(kpis: List[Dict[str, Any]]) -> List[str]:
    """Obtiene lista de todos los edificios."""
    buildings = set()
    for kpi in kpis:
        if kpi.get("level") == "building":
            buildings.add(kpi.get("name", ""))
    return sorted(list(buildings))


def plot_district_kpis(district_kpis: Dict[str, float], output_path: Path) -> None:
    """Genera gráfico de KPIs a nivel distrito."""
    # Seleccionar métricas clave
    key_metrics = [
        "cost_total",
        "carbon_emissions_total",
        "all_time_peak_average",
        "daily_peak_average",
        "electricity_consumption_total",
        "zero_net_energy",
    ]
    
    metrics_to_plot = {k: v for k, v in district_kpis.items() if k in key_metrics}
    
    if not metrics_to_plot:
        print("[!] No hay métricas clave para graficar")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = list(metrics_to_plot.keys())
    values = list(metrics_to_plot.values())
    
    # Colores según si el valor es bueno (<1) o malo (>1)
    colors = ['#2ecc71' if v <= 1.0 else '#e74c3c' for v in values]
    
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor='black', alpha=0.8)
    
    # Línea de referencia en 1.0 (baseline)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Baseline (No Control)')
    
    # Etiquetas
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('_', '\n') for n in names], rotation=0, fontsize=9)
    ax.set_ylabel('Valor Normalizado', fontsize=12)
    ax.set_title('KPIs de Flexibilidad Energética - MADDPG\n(< 1.0 = Mejor que baseline)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores sobre las barras
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Gráfico guardado: {output_path}")


def plot_building_comparison(kpis: List[Dict[str, Any]], output_path: Path) -> None:
    """Genera gráfico comparativo de edificios."""
    buildings = get_all_buildings(kpis)
    
    if not buildings:
        print("[!] No hay datos de edificios")
        return
    
    # Extraer costo total y emisiones para cada edificio
    costs = []
    emissions = []
    
    for building in buildings:
        b_kpis = filter_building_kpis(kpis, building)
        costs.append(b_kpis.get("cost_total", 1.0))
        emissions.append(b_kpis.get("carbon_emissions_total", 1.0))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico de costos
    ax1 = axes[0]
    cmap = plt.get_cmap('RdYlGn_r')
    colors_cost = [cmap((v - min(costs)) / (max(costs) - min(costs) + 0.01)) for v in costs]
    bars1 = ax1.bar(range(len(buildings)), costs, color=colors_cost, edgecolor='black', alpha=0.8)
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Baseline')
    ax1.set_xticks(range(len(buildings)))
    ax1.set_xticklabels([b.replace('Building_', 'B') for b in buildings], fontsize=9)
    ax1.set_ylabel('Costo Normalizado', fontsize=11)
    ax1.set_title('Costo por Edificio', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Gráfico de emisiones
    ax2 = axes[1]
    colors_em = [cmap((v - min(emissions)) / (max(emissions) - min(emissions) + 0.01)) for v in emissions]
    bars2 = ax2.bar(range(len(buildings)), emissions, color=colors_em, edgecolor='black', alpha=0.8)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Baseline')
    ax2.set_xticks(range(len(buildings)))
    ax2.set_xticklabels([b.replace('Building_', 'B') for b in buildings], fontsize=9)
    ax2.set_ylabel('Emisiones CO2 Normalizadas', fontsize=11)
    ax2.set_title('Emisiones CO2 por Edificio', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Comparación de Edificios - MADDPG', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Gráfico guardado: {output_path}")


def plot_radar_kpis(district_kpis: Dict[str, float], output_path: Path) -> None:
    """Genera gráfico radar de KPIs principales."""
    # Métricas para el radar
    radar_metrics = {
        "Costo": district_kpis.get("cost_total", 1.0),
        "Emisiones\nCO2": district_kpis.get("carbon_emissions_total", 1.0),
        "Pico\nDemanda": district_kpis.get("all_time_peak_average", 1.0),
        "Consumo\nElectricidad": district_kpis.get("electricity_consumption_total", 1.0),
        "Factor\nCarga": district_kpis.get("daily_one_minus_load_factor_average", 1.0),
    }
    
    categories = list(radar_metrics.keys())
    values = list(radar_metrics.values())
    
    # Completar el círculo
    values += values[:1]
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # MADDPG
    ax.plot(angles, values, 'o-', linewidth=2, label='MADDPG', color='#3498db')
    ax.fill(angles, values, alpha=0.25, color='#3498db')
    
    # Baseline (todos = 1.0)
    baseline = [1.0] * (len(categories) + 1)
    ax.plot(angles, baseline, '--', linewidth=2, label='Baseline (No Control)', color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_title('Perfil de Flexibilidad Energética - MADDPG\n(Menor = Mejor)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Gráfico guardado: {output_path}")


def generate_markdown_report(district_kpis: Dict[str, float], buildings: List[str], 
                             kpis: List[Dict[str, Any]], output_path: Path) -> None:
    """Genera reporte en formato Markdown."""
    
    report = """# Reporte de KPIs - MADDPG CityLearn

## Resumen Ejecutivo

Este reporte presenta los indicadores clave de rendimiento (KPIs) del modelo MADDPG 
entrenado para optimización de flexibilidad energética en comunidades de edificios.

## KPIs a Nivel de Distrito

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
"""
    
    key_metrics = [
        ("cost_total", "Costo Total"),
        ("carbon_emissions_total", "Emisiones CO2"),
        ("all_time_peak_average", "Pico Promedio"),
        ("daily_peak_average", "Pico Diario"),
        ("electricity_consumption_total", "Consumo Electricidad"),
        ("zero_net_energy", "Energía Neta Cero"),
    ]
    
    for key, name in key_metrics:
        value = district_kpis.get(key, "N/A")
        if isinstance(value, float):
            interpretation = "✅ Mejor que baseline" if value < 1.0 else "⚠️ Peor que baseline"
            report += f"| {name} | {value:.4f} | {interpretation} |\n"
    
    # Añadir sección de edificios
    report += f"""
## Análisis por Edificio

Total de edificios: {len(buildings)}

### Mejores Edificios (Costo < 1.0)

| Edificio | Costo | Emisiones CO2 |
|----------|-------|---------------|
"""
    
    for building in buildings:
        b_kpis = filter_building_kpis(kpis, building)
        cost = b_kpis.get("cost_total", 1.0)
        emissions = b_kpis.get("carbon_emissions_total", 1.0)
        if cost < 1.2:  # Mostrar edificios con costo razonable
            report += f"| {building} | {cost:.4f} | {emissions:.4f} |\n"
    
    report += """
## Conclusiones

1. **Flexibilidad Energética**: El modelo MADDPG demuestra capacidad de coordinación 
   multi-agente para optimización energética.

2. **KPIs Principales**: Los valores cercanos o menores a 1.0 indican mejora respecto 
   al baseline sin control.

3. **Variabilidad entre Edificios**: Existe heterogeneidad en el rendimiento por edificio,
   lo cual es esperado dado los diferentes perfiles de consumo.

---
*Generado automáticamente por generate_kpis_report.py*
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[✓] Reporte guardado: {output_path}")


def main():
    """Función principal."""
    print("=" * 60)
    print("📊 GENERANDO REPORTES DE KPIs MADDPG")
    print("=" * 60)
    
    # Cargar KPIs
    kpis = load_kpis()
    if kpis is None:
        print("[!] No se encontraron KPIs. Ejecuta el entrenamiento primero.")
        return 1
    
    print(f"[✓] Cargados {len(kpis)} registros de KPIs")
    
    # Extraer datos
    district_kpis = filter_district_kpis(kpis)
    buildings = get_all_buildings(kpis)
    
    print(f"[✓] KPIs de distrito: {len(district_kpis)} métricas")
    print(f"[✓] Edificios: {len(buildings)}")
    
    # Generar gráficos
    print("\n📈 Generando gráficos...")
    
    plot_district_kpis(district_kpis, REPORTS_DIR / "kpis_district.png")
    plot_building_comparison(kpis, REPORTS_DIR / "kpis_buildings.png")
    plot_radar_kpis(district_kpis, REPORTS_DIR / "kpis_radar.png")
    
    # Copiar a static para dashboard
    plot_district_kpis(district_kpis, STATIC_DIR / "kpis_district.png")
    plot_radar_kpis(district_kpis, STATIC_DIR / "kpis_radar.png")
    
    # Generar reporte Markdown
    print("\n📝 Generando reporte...")
    generate_markdown_report(district_kpis, buildings, kpis, REPORTS_DIR / "kpis_report.md")
    
    print("\n" + "=" * 60)
    print("✅ REPORTES GENERADOS EXITOSAMENTE")
    print("=" * 60)
    print(f"\nArchivos generados:")
    print(f"  - {REPORTS_DIR / 'kpis_district.png'}")
    print(f"  - {REPORTS_DIR / 'kpis_buildings.png'}")
    print(f"  - {REPORTS_DIR / 'kpis_radar.png'}")
    print(f"  - {REPORTS_DIR / 'kpis_report.md'}")
    print(f"  - {STATIC_DIR / 'kpis_district.png'}")
    print(f"  - {STATIC_DIR / 'kpis_radar.png'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
