"""Script para inspeccionar recursos específicos de cada edificio en CityLearn v2"""

from citylearn.citylearn import CityLearnEnv

# Cargar el entorno
env = CityLearnEnv(
    schema="citylearn_challenge_2022_phase_all_plus_evs", central_agent=False
)

print(f"Total de edificios: {len(env.buildings)}\n")
print("=" * 120)

for i, building in enumerate(env.buildings):
    print(f"\n🏢 Edificio {i}: {building.name}")
    print("-" * 120)

    # Solar PV
    if building.pv is not None and building.pv.nominal_power is not None:
        print(
            f"  ☀️  Solar PV: {building.pv.nominal_power:.2f} kW (capacidad nominal)"
        )
    else:
        print(f"  ☀️  Solar PV: NO disponible")

    # Batería eléctrica
    if building.electrical_storage is not None:
        capacity = building.electrical_storage.capacity
        efficiency = building.electrical_storage.efficiency
        max_power = building.electrical_storage.nominal_power
        print(
            f"  🔋 Batería: {capacity:.2f} kWh | Eficiencia: {efficiency:.2%} | Max: {max_power:.2f} kW"
        )
    else:
        print(f"  🔋 Batería: NO disponible")

    # Vehículo eléctrico
    if (
        hasattr(building, "electric_vehicle")
        and building.electric_vehicle is not None
    ):
        ev = building.electric_vehicle
        ev_capacity = ev.capacity if hasattr(ev, "capacity") else "N/A"
        ev_charger = (
            ev.charger_power if hasattr(ev, "charger_power") else "N/A"
        )
        print(f"  🚗 EV: Batería {ev_capacity} kWh | Cargador: {ev_charger} kW")
    else:
        print(f"  🚗 EV: NO disponible")

    # HVAC (Cooling + Heating)
    cooling = "NO disponible"
    heating = "NO disponible"

    if building.cooling_device is not None:
        cooling_power = building.cooling_device.nominal_power
        cooling = (
            f"{cooling_power:.2f} kW"
            if cooling_power is not None
            else "Disponible"
        )

    if building.heating_device is not None:
        heating_power = building.heating_device.nominal_power
        heating = (
            f"{heating_power:.2f} kW"
            if heating_power is not None
            else "Disponible"
        )

    print(f"  ❄️  HVAC Cooling: {cooling}")
    print(f"  🔥 HVAC Heating: {heating}")

    # DHW (Domestic Hot Water)
    if building.dhw_storage is not None:
        dhw_capacity = building.dhw_storage.capacity
        print(f"  🚿 DHW Storage: {dhw_capacity:.2f} kWh")
    else:
        print(f"  🚿 DHW: NO disponible")

    # Dimensiones
    obs = building.observations()
    actions = building.action_space
    print(f"  📊 Observaciones: {len(obs)} dimensiones")
    print(f"  🎮 Acciones: {actions.shape[0]} dimensiones")

print("\n" + "=" * 120)
print("\n✅ Inspección completada")
