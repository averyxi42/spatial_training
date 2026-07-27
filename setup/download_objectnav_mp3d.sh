#!/usr/bin/env bash
# Download and extract the ObjectNav MP3D v1 episode dataset into
# data/datasets/objectnav/mp3d/v1/ (paths relative to the repo root).
set -euo pipefail

URL="https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${REPO_ROOT}/data/datasets/objectnav/mp3d/v1"
ZIP_PATH="${DEST_DIR}/objectnav_mp3d_v1.zip"

mkdir -p "${DEST_DIR}"

echo "Downloading ${URL}"
echo "     -> ${ZIP_PATH}"
# -C - resumes a partial download if the script is re-run.
curl -L -C - --fail --retry 3 -o "${ZIP_PATH}" "${URL}"

echo "Extracting into ${DEST_DIR}"
unzip -o -q "${ZIP_PATH}" -d "${DEST_DIR}"

# The archive contains a top-level objectnav_mp3d_v1/ folder; flatten it so the
# episode files land directly in v1/.
if [ -d "${DEST_DIR}/objectnav_mp3d_v1" ]; then
  cp -rn "${DEST_DIR}/objectnav_mp3d_v1/." "${DEST_DIR}/"
  rm -rf "${DEST_DIR}/objectnav_mp3d_v1"
fi

rm -f "${ZIP_PATH}"
echo "Done. Contents of ${DEST_DIR}:"
ls -1 "${DEST_DIR}"
