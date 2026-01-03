"""Script detallado para inspeccionar si realmente hay EVs en el dataset"""

from citylearn.citylearn import CityLearnEnv
import numpy as np

# Cargar el entorno
env = CityLearnEnv(
    schema="citylearn_challenge_2022_phase_all_plus_evs", central_agent=False
)

print("=" * 120)
print("INSPECCIÓN DETALLADA DE VEHÍCULOS ELÉCTRICOS (EVs)")
print("=" * 120)

# Reset para obtener observaciones iniciales
observations, _ = env.reset()

print(f"\nTotal de edificios: {len(env.buildings)}")
print("\n" + "=" * 120)

has_ev_data = False

for i, building in enumerate(env.buildings):
    print(f"\n🏢 Edificio {i}: {building.name}")
    print("-" * 120)

    # Verificar si tiene atributo electric_vehicle
    has_ev_attr = hasattr(building, "electric_vehicle")
    print(f"  ├─ Atributo 'electric_vehicle': {has_ev_attr}")

    if has_ev_attr and building.electric_vehicle is not None:  # type: ignore[attr-defined]
        ev = building.electric_vehicle  # type: ignore[attr-defined]
        print(f"  ├─ EV object: {type(ev).__name__}")

        # Inspeccionar atributos del EV
        if hasattr(ev, "capacity"):
            print(f"  ├─ EV Capacidad batería: {ev.capacity} kWh")
        if hasattr(ev, "charger_power"):
            print(f"  ├─ EV Potencia cargador: {ev.charger_power} kW")
        if hasattr(ev, "efficiency"):
            print(f"  ├─ EV Eficiencia: {ev.efficiency}")

    # Verificar observaciones relacionadas con EV
    obs_dict = building.observations()
    ev_obs_names = [
        "electric_vehicle_arrival",
        "electric_vehicle_availability",
        "electric_vehicle_charge_rate",
        "electric_vehicle_energy_charged",
        "electric_vehicle_state_of_charge",
    ]

    print("  ├─ Observaciones EV disponibles:")
    for ev_obs_name in ev_obs_names:
        if ev_obs_name in obs_dict:
            value = obs_dict[ev_obs_name]
            print(f"  │  ├─ {ev_obs_name}: {value}")
            if value is not None and not (
                isinstance(value, (int, float)) and value == 0
            ):
                has_ev_data = True
        else:
            print(f"  │  ├─ {ev_obs_name}: NO DISPONIBLE")

    # Verificar action space para EV
    action_space = building.action_space
    action_names = building.action_metadata
    print(f"  └─ Acciones disponibles: {action_names}")

print("\n" + "=" * 120)

# Simular varios timesteps para ver si hay datos EV reales
print("\n📊 SIMULACIÓN DE 100 TIMESTEPS PARA DETECTAR DATOS EV REALES")
print("=" * 120)

ev_data_detected = {i: False for i in range(len(env.buildings))}
non_zero_values = {i: [] for i in range(len(env.buildings))}

for step in range(100):
    # Acciones aleatorias
    actions = [
        np.random.uniform(-1, 1, size=space.shape).tolist()
        for space in env.action_space
    ]

    try:
        observations, rewards, info, terminated, truncated = env.step(actions)  # type: ignore[arg-type]

        for i, obs in enumerate(observations):
            building = env.buildings[i]
            obs_dict = building.observations()

            # Verificar si hay valores no-cero en observaciones EV
            ev_availability = obs_dict.get("electric_vehicle_availability", 0)
            ev_soc = obs_dict.get("electric_vehicle_state_of_charge", 0)
            ev_charge_rate = obs_dict.get("electric_vehicle_charge_rate", 0)

            if ev_availability > 0 or ev_soc > 0 or ev_charge_rate > 0:
                ev_data_detected[i] = True
                if step < 10:  # Solo guardar primeros 10 para no saturar
                    non_zero_values[i].append(
                        {
                            "step": step,
                            "availability": ev_availability,
                            "soc": ev_soc,
                            "charge_rate": ev_charge_rate,
                        }
                    )
    except Exception:
        break

print("\n🔍 RESULTADOS DE LA DETECCIÓN:")
print("-" * 120)

total_with_ev_data = sum(ev_data_detected.values())

if total_with_ev_data > 0:
    print(f"✅ SE DETECTARON DATOS EV REALES en {total_with_ev_data} edificios")
    print("\nEdificios con datos EV:")
    for i, has_data in ev_data_detected.items():
        if has_data:
            print(f"  - Building_{i+1} (ID {i})")
            if non_zero_values[i]:
                print("    Ejemplos de valores no-cero:")
                for val in non_zero_values[i][:3]:
                    print(
                        (
                            "      Step {step}: availability={availability}, "
                            "soc={soc:.3f}, charge_rate={charge_rate:.3f}"
                        ).format(
                            step=val["step"],
                            availability=val["availability"],
                            soc=val["soc"],
                            charge_rate=val["charge_rate"],
                        )
                    )
else:
    print("❌ NO SE DETECTARON DATOS EV REALES en ningún edificio")
    print("\n⚠️ CONCLUSIÓN: Las observaciones EV existen en el schema pero:")
    print("   1. Todos los valores son 0 o None")
    print("   2. No hay vehículos eléctricos realmente modelados")
    print("   3. El nombre 'plus_evs' es engañoso")

print("\n" + "=" * 120)
print("✅ Inspección completada")
