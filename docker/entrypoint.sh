#!/usr/bin/env bash
# Activates the `vln` conda env, then execs whatever the container was given.
set -euo pipefail
echo "initializaing"
# shellcheck disable=SC1091
source "${CONDA_DIR:-/opt/conda}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-vln}"

# The repo is bind-mounted, so its editable install cannot be baked into the
# image. Install it once per container unless LONGNAV_SKIP_INSTALL is set.
if [ -z "${LONGNAV_SKIP_INSTALL:-}" ] && [ -f /workspace/pyproject.toml ]; then
    if ! python -c "import longnav" >/dev/null 2>&1; then
        echo "[entrypoint] installing longnav (and verl, if present) in editable mode"
        if [ -f /workspace/verl/setup.py ] || [ -f /workspace/verl/pyproject.toml ]; then
            pip install --no-dependencies -e /workspace/verl >/dev/null
        fi
        pip install --no-dependencies -e /workspace >/dev/null
    fi
fi
echo "activating env"
conda deactivate
conda activate longnav
exec "$@"
