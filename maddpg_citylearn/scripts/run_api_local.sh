#!/usr/bin/env bash
# TODO: pegar aquí el contenido final de run_api_local.sh
#!/usr/bin/env bash
export PYTHONPATH=src:${PYTHONPATH}
uvicorn maddpg_tesis.api.main:app --host 0.0.0.0 --port 8000 --reload
