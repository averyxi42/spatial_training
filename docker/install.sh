#!/usr/bin/env bash
# Install LongNav-R1 into both conda envs. Used for the docker container and for
# a bare-machine install alike.
#
#   docker/install.sh [--repo PATH]
#   INSTALL_FLASH_ATTN=1 docker/install.sh      # adds flash-attn to the VLM env
#
# There are two envs because the code requires them: config_schema.py defaults
# habitat_conda_env="vln" / vlm_conda_env="longnav", and factories.py passes
# those to Ray as runtime_env={"conda": ...}. Ray re-execs each actor inside the
# named env, so both need the project installed and a matching `ray` — an env
# missing it fails at actor startup rather than at import. The habitat env gets
# the bare core (ray + hydra-core); only the VLM env needs the [vlm] extra.
#
# Env names can be overridden with HABITAT_ENV_NAME / VLM_ENV_NAME, but they must
# then match the config, or Ray will look for envs that do not exist.
set -euo pipefail

VLM_ENV_NAME="${VLM_ENV_NAME:-longnav}"
HABITAT_ENV_NAME="${HABITAT_ENV_NAME:-vln}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
REPO=""
# take repo as CLI arg
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo)
            REPO="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done
# Default to the repo this script lives in.
if [ -z "${REPO}" ]; then
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [ ! -f "${REPO}/pyproject.toml" ]; then
    echo "no pyproject.toml at ${REPO} — pass --repo <path>" >&2
    exit 1
fi

pip_habitat() { conda run --no-capture-output -n "${HABITAT_ENV_NAME}" pip "$@"; }
pip_vlm() { conda run --no-capture-output -n "${VLM_ENV_NAME}" pip "$@"; }

echo "=== longnav install: habitat_env=${HABITAT_ENV_NAME} vlm_env=${VLM_ENV_NAME} repo=${REPO}"
echo "--- installing ray interface into habitat env"
pip_habitat install -e "${REPO}"  # install the repo into the habitat env, editable

# habitat-sim's magnum bindings are compiled against the numpy C API present at
# build time, so a numpy that moved would break `import habitat_sim` with no
# error at install time. The core deps above should not touch numpy; warn rather
# than reinstall, so a deliberate change is not silently undone.
HABITAT_NUMPY="$(conda run -n "${HABITAT_ENV_NAME}" python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo missing)"
if [ "${HABITAT_NUMPY}" != "1.26.4" ]; then
    echo "!!! WARNING: numpy in ${HABITAT_ENV_NAME} is ${HABITAT_NUMPY}, expected 1.26.4."
    echo "!!! habitat_sim is compiled against 1.26.4 and will likely fail to import."
fi

if [ -f "${REPO}/verl/pyproject.toml" ]; then
    echo "--- verl (editable, no deps)"
    pip_vlm install --no-dependencies -e "${REPO}/verl"
else
    echo "--- verl not found at ${REPO}/verl, skipping (git submodule update --init)"
fi
echo "--- installing longnav into vlm env"
pip_vlm install -e "${REPO}[vlm]"  # install the repo into the VLM env, editable

# flash-attn has to come after the [vlm] extra: its setup.py imports torch, and
# --no-build-isolation means it uses the env's torch rather than fetching its
# own. That is also why it cannot live in the Dockerfile, which installs no
# python packages. Slow (a source build), hence opt-in.
if [ "${INSTALL_FLASH_ATTN}" != "0" ]; then
    echo "--- flash-attn (source build, slow)"
    pip_vlm install flash-attn --no-build-isolation
fi
