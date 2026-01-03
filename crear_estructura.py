import os
from pathlib import Path

# Carpeta raíz del proyecto
ROOT = Path("maddpg_citylearn")

# Directorios que queremos crear
DIRS = [
    ROOT / "src" / "maddpg_tesis" / "core",
    ROOT / "src" / "maddpg_tesis" / "envs",
    ROOT / "src" / "maddpg_tesis" / "maddpg",
    ROOT / "src" / "maddpg_tesis" / "models",
    ROOT / "src" / "maddpg_tesis" / "api",
    ROOT / "configs",
    ROOT / "scripts",
    ROOT / "models",  # para checkpoints entrenados
]

# Archivos que queremos crear (ruta -> contenido inicial)
FILES = {
    # Paquete principal
    ROOT / "src" / "maddpg_tesis" / "__init__.py": '"""Paquete MADDPG CityLearn (MADRL)."""\n',

    # core
    ROOT / "src" / "maddpg_tesis" / "core" / "__init__.py": '"""Módulos centrales (config, logging, utils)."""\n',
    ROOT / "src" / "maddpg_tesis" / "core" / "config.py": "# TODO: pegar aquí el código de config.py\n",
    ROOT / "src" / "maddpg_tesis" / "core" / "logging.py": "# TODO: pegar aquí el código de logging.py\n",
    ROOT / "src" / "maddpg_tesis" / "core" / "utils.py": "# TODO: pegar aquí el código de utils.py\n",

    # envs
    ROOT / "src" / "maddpg_tesis" / "envs" / "__init__.py": '"""Wrappers de entornos (CityLearn v2, etc.)."""\n',
    ROOT / "src" / "maddpg_tesis" / "envs" / "citylearn_env.py": "# TODO: pegar aquí el código de citylearn_env.py\n",

    # maddpg
    ROOT / "src" / "maddpg_tesis" / "maddpg" / "__init__.py": '"""Implementación de MADDPG para MADRL.""\"\n',
    ROOT / "src" / "maddpg_tesis" / "maddpg" / "policies.py": "# TODO: pegar aquí el código de policies.py\n",
    ROOT / "src" / "maddpg_tesis" / "maddpg" / "noise.py": "# TODO: pegar aquí el código de noise.py\n",
    ROOT / "src" / "maddpg_tesis" / "maddpg" / "replay_buffer.py": "# TODO: pegar aquí el código de replay_buffer.py\n",
    ROOT / "src" / "maddpg_tesis" / "maddpg" / "agent.py": "# TODO: pegar aquí el código de agent.py\n",
    ROOT / "src" / "maddpg_tesis" / "maddpg" / "maddpg.py": "# TODO: pegar aquí el código de maddpg.py\n",

    # models
    ROOT / "src" / "maddpg_tesis" / "models" / "__init__.py": '"""Utilidades para guardar/cargar modelos MADDPG."""\n',
    ROOT / "src" / "maddpg_tesis" / "models" / "loader.py": "# TODO: pegar aquí el código de loader.py\n",

    # api
    ROOT / "src" / "maddpg_tesis" / "api" / "__init__.py": '"""API REST (FastAPI) para inferencia MADDPG CityLearn.""\"\n',
    ROOT / "src" / "maddpg_tesis" / "api" / "schemas.py": "# TODO: pegar aquí el código de schemas.py\n",
    ROOT / "src" / "maddpg_tesis" / "api" / "deps.py": "# TODO: pegar aquí el código de deps.py\n",
    ROOT / "src" / "maddpg_tesis" / "api" / "main.py": "# TODO: pegar aquí el código de main.py\n",

    # configs
    ROOT / "configs" / "citylearn_maddpg.yaml": "# TODO: pegar aquí la config YAML.\n",

    # scripts
    ROOT / "scripts" / "train_citylearn.py": "# TODO: pegar aquí el código de train_citylearn.py\n",
    ROOT / "scripts" / "run_api_local.sh": "#!/usr/bin/env bash\n# TODO: pegar aquí el contenido final de run_api_local.sh\n",

    # raíz
    ROOT / "requirements.txt": "# TODO: pegar aquí las dependencias (requirements.txt)\n",
    ROOT / "Dockerfile": "# TODO: pegar aquí el contenido del Dockerfile\n",
    ROOT / "README.md": "# Proyecto MADDPG CityLearn (MADRL)\n\nDescribe aquí el proyecto para tu tesis.\n",
}


def main():
    # Crear directorios
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Directorio creado (o ya existía): {d}")

    # Crear archivos
    for path, content in FILES.items():
        if not path.exists():
            with path.open("w", encoding="utf-8") as f:
                f.write(content)
            print(f"Archivo creado: {path}")
        else:
            print(f"Archivo ya existe, no se sobrescribe: {path}")

    # Dar permiso de ejecución al script .sh en Unix
    run_sh = ROOT / "scripts" / "run_api_local.sh"
    if run_sh.exists():
        try:
            mode = run_sh.stat().st_mode
            run_sh.chmod(mode | 0o111)  # añadir permisos de ejecución
            print(f"Permisos de ejecución añadidos a: {run_sh}")
        except Exception as e:
            print(f"No se pudo cambiar permisos de {run_sh}: {e}")

    print("\nEstructura de proyecto creada en:", ROOT.resolve())


if __name__ == "__main__":
    main()
