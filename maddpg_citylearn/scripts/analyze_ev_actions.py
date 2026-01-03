"""Script para analizar las acciones EV disponibles"""

from citylearn.citylearn import CityLearnEnv

env = CityLearnEnv(
    schema="citylearn_challenge_2022_phase_all_plus_evs", central_agent=False
)

print("=" * 120)
print("ANÁLISIS DE CARGADORES EV COMO ACCIONES")
print("=" * 120)

buildings_with_ev_chargers = []

for i, building in enumerate(env.buildings):
    action_metadata = building.action_metadata
    action_space_dim = building.action_space.shape[0]

    # Buscar acciones relacionadas con EV
    ev_actions = [
        key
        for key in action_metadata.keys()
        if "electric_vehicle" in key.lower()
    ]

    if ev_actions:
        buildings_with_ev_chargers.append(i)
        print(f"\n🏢 Building_{i + 1} (ID {i})")
        print(f"  ├─ Dimensión action space: {action_space_dim}")
        print(f"  ├─ Total acciones disponibles: {len(action_metadata)}")
        print(f"  ├─ Acciones EV charger: {ev_actions}")
        print(f"  └─ Todas las acciones: {list(action_metadata.keys())}")

print("\n" + "=" * 120)
print("\n📊 RESUMEN:")
print(f"  - Total edificios: {len(env.buildings)}")
print(f"  - Edificios CON cargadores EV: {len(buildings_with_ev_chargers)}")
print(f"  - IDs con cargadores: {buildings_with_ev_chargers}")

print("\n⚠️ OBSERVACIÓN CRÍTICA:")
print("  Los cargadores EV están disponibles como ACCIONES (control de carga)")
print(
    "  PERO las OBSERVACIONES de estado del EV (SoC, availability) "
    "NO están disponibles"
)
print("  Esto significa:")
print("    - ✅ Puedes CONTROLAR la carga del EV")
print("    - ❌ NO puedes VER el estado del EV (SoC, conectado, etc.)")
print("    - 🤔 Es un control 'a ciegas' o basado en cargas fijas")

print("\n" + "=" * 120)

# Verificar dimensiones reales
print("\n📏 DIMENSIONES REALES POR EDIFICIO:")
print("=" * 120)

for i, building in enumerate(env.buildings):
    obs_dim = len(building.observations())
    action_dim = building.action_space.shape[0]
    action_metadata = building.action_metadata

    ev_chargers = [
        key
        for key in action_metadata.keys()
        if "electric_vehicle" in key.lower()
    ]
    washing_machines = [
        key
        for key in action_metadata.keys()
        if "washing_machine" in key.lower()
    ]

    active_actions = []
    if action_metadata.get("electrical_storage", False):
        active_actions.append("battery")
    if action_metadata.get("cooling_storage", False):
        active_actions.append("cooling")
    if action_metadata.get("dhw_storage", False):
        active_actions.append("dhw")
    if ev_chargers:
        active_actions.append(f"ev_charger×{len(ev_chargers)}")
    if washing_machines:
        active_actions.append(f"washing×{len(washing_machines)}")

    print(
        (
            f"Building_{i + 1:2d}: obs={obs_dim:2d}, "
            f"actions={action_dim}, recursos={active_actions}"
        )
    )

print("\n" + "=" * 120)
