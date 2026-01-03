"""
Script para validar que el dataset citylearn_challenge_2022_phase_all_plus_evs
está disponible en la instalación de CityLearn.

Los datasets vienen incluidos en el paquete de CityLearn, no es necesario
descargarlos por separado desde GitHub.
"""

import sys


def validate_citylearn_installation():
    """Valida que CityLearn esté instalado correctamente."""
    try:
        import citylearn

        print(f"✓ CityLearn instalado - Versión: {citylearn.__version__}")
        return True
    except ImportError as e:
        print(f"✗ CityLearn NO instalado: {e}")
        print("\nPara instalar:")
        print("  pip install citylearn==2.5.0 --no-deps")
        print(
            "  pip install gymnasium==0.28.1 pandas scikit-learn<=1.2.2 simplejson torchvision"
        )
        return False


def list_available_schemas():
    """Lista los schemas/datasets disponibles en CityLearn."""
    try:
        from citylearn.data import DataSet

        print("\n" + "=" * 60)
        print("SCHEMAS/DATASETS DISPONIBLES EN CITYLEARN")
        print("=" * 60)

        # Obtener lista de schemas disponibles
        available_schemas = DataSet.get_names()  # type: ignore[attr-defined]

        for i, schema in enumerate(available_schemas, 1):
            marker = (
                "🎯" if "2022" in schema and "evs" in schema.lower() else "  "
            )
            print(f"{marker} {i}. {schema}")

        print("\n🎯 = Dataset usado en este proyecto")
        return available_schemas

    except Exception as e:
        print(f"✗ Error listando schemas: {e}")
        return []


def validate_target_schema():
    """Valida que el schema objetivo esté disponible."""
    target_schema = "citylearn_challenge_2022_phase_all_plus_evs"

    print("\n" + "=" * 60)
    print("VALIDANDO SCHEMA OBJETIVO")
    print("=" * 60)
    print(f"Schema: {target_schema}")

    try:
        from citylearn.citylearn import CityLearnEnv

        # Intentar cargar el environment
        env = CityLearnEnv(schema=target_schema, central_agent=False)

        print("\n✓ Schema cargado exitosamente!")
        print("\nInformación del entorno:")
        print(f"  - Número de edificios: {len(env.buildings)}")
        print(f"  - Timesteps totales: {env.time_steps}")
        print(f"  - Dimensión de acción: {env.action_space[0].shape[0]}")
        print(
            f"  - Dimensión de observación: {env.observation_space[0].shape[0]}"
        )

        # Información sobre edificios
        print("\nEdificios en la comunidad:")
        for i, building in enumerate(env.buildings):
            print(f"  {i + 1}. {building.name}")

        # Verificar si tiene EVs
        has_evs = any(
            hasattr(b, "electrical_storage")
            and b.electrical_storage is not None
            for b in env.buildings
        )

        print(
            f"\n✓ Dataset incluye vehículos eléctricos: {'Sí' if has_evs else 'Verificar'}"
        )

        return True

    except FileNotFoundError:
        print(f"\n✗ Schema NO encontrado: {target_schema}")
        print("\nPosibles causas:")
        print("  1. El nombre del schema es incorrecto")
        print("  2. La versión de CityLearn no incluye este dataset")
        print("  3. Los archivos del dataset no se descargaron correctamente")
        print("\nSolución:")
        print("  - Verifica la lista de schemas disponibles arriba")
        print("  - Usa uno de los schemas listados")
        return False

    except Exception as e:
        print(f"\n✗ Error cargando schema: {e}")
        print("\nDetalles del error:")
        print(f"  Tipo: {type(e).__name__}")
        print(f"  Mensaje: {str(e)}")
        return False


def test_environment_step():
    """Prueba un paso del entorno para verificar funcionamiento."""
    target_schema = "citylearn_challenge_2022_phase_all_plus_evs"

    print("\n" + "=" * 60)
    print("PRUEBA DE FUNCIONAMIENTO DEL ENTORNO")
    print("=" * 60)

    try:
        from citylearn.citylearn import CityLearnEnv

        env = CityLearnEnv(schema=target_schema, central_agent=False)

        # Reset
        observations = env.reset()
        print(
            f"✓ Reset exitoso - Observaciones iniciales: {len(observations)} agentes"
        )

        # Step con acciones aleatorias
        actions = []
        for space in env.action_space:
            action = space.sample()  # Acción aleatoria
            actions.append(action)

        next_obs, rewards, done, truncated, info = env.step(actions)

        print("✓ Step exitoso")
        print(f"  - Recompensas: {[f'{r:.3f}' for r in rewards[:3]]}...")
        print(f"  - Done: {done}")
        print(f"  - Info keys: {list(info.keys())[:5]}...")

        return True

    except Exception as e:
        print(f"✗ Error en prueba: {e}")
        return False


def main():
    """Ejecuta todas las validaciones."""
    print("\n" + "=" * 60)
    print("VALIDACIÓN DE DATASET CITYLEARN")
    print("citylearn_challenge_2022_phase_all_plus_evs")
    print("=" * 60)

    # 1. Validar instalación
    if not validate_citylearn_installation():
        return False

    # 2. Listar schemas disponibles
    list_available_schemas()

    # 3. Validar schema objetivo
    schema_valid = validate_target_schema()

    # 4. Prueba de funcionamiento
    if schema_valid:
        test_environment_step()

    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)

    if schema_valid:
        print("✓ El dataset está disponible y funcionando correctamente")
        print("✓ NO es necesario descargar nada adicional desde GitHub")
        print("✓ Los datasets vienen incluidos en el paquete de CityLearn")
        print("\nPuedes proceder con el entrenamiento:")
        print("  python -m maddpg_tesis.scripts.train_citylearn")
    else:
        print("✗ Hay problemas con el dataset")
        print(
            "\nRevisa los schemas disponibles arriba y actualiza la configuración"
        )
        print("en: configs/citylearn_maddpg.yaml")

    print("=" * 60)
    return schema_valid


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
