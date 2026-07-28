#!/usr/bin/env bash
# Minimal by design. This does only what cannot be done at build time — adapt to
# whatever uid the container was handed — and then gets out of the way.
#
# It installs nothing — that is docker/install.sh, run explicitly. Doing it here
# would re-run on every container start and be discarded again under
# `docker compose run --rm`, which throws away the writable layer on exit.
# Deliberately no `set -u`: conda's shell functions reference unset variables
# and would abort the entrypoint on activation.
set -eo pipefail

CONDA_DIR="${CONDA_DIR:-/opt/conda}"

# The image carries no baked-in uid, so compose can pass any `user:` without a
# rebuild. But a uid with no passwd entry breaks sudo ("you do not exist in the
# passwd database"), `id -gn`, and anything that resolves $HOME. /etc/passwd and
# /etc/group are mode 666 from the build so we can fill the gap here.
if ! getent passwd "$(id -u)" >/dev/null 2>&1; then
    echo "longnav:x:$(id -u):$(id -g)::/home/longnav:/bin/bash" >> /etc/passwd 2>/dev/null || true
fi
if ! getent group "$(id -g)" >/dev/null 2>&1; then
    echo "longnav:x:$(id -g):" >> /etc/group 2>/dev/null || true
fi

export HOME=/home/longnav
[ -w "${HOME}" ] || export HOME=/tmp

# LONGNAV_ENV picks which env the shell lands in; both stay on PATH either way,
# which matters because Ray activates the other one by name for its actors.
# shellcheck disable=SC1091
source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${LONGNAV_ENV:-longnav}"

exec "$@"
