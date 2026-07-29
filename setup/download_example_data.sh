#!/usr/bin/env bash
# Download the single MP3D example scene (17DRP5sb8fy, ~93 MB) — enough to run
# tests/mp3d_example_env.py without the full ~15 GB MP3D set.
#
# The episode dataset still comes from setup/download_objectnav_mp3d.sh; the test
# config restricts it to this one scene via habitat's `content_scenes`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

conda run --no-capture-output -n "${HABITAT_ENV_NAME:-vln}" \
    python -m habitat_sim.utils.datasets_download \
        --uids mp3d_example_scene --data-path data/

cd data/scene_datasets

# The downloader writes mp3d_example as an *absolute* symlink, which breaks the
# moment the repo is seen at a different path — most obviously inside the docker
# container, where it is bind-mounted at /workspace. Rewrite it relative.
if [ -L mp3d_example ] && [ -d "${REPO_ROOT}/data/versioned_data/mp3d_example_scene_1.1" ]; then
    ln -sfn ../versioned_data/mp3d_example_scene_1.1 mp3d_example
    echo "relinked mp3d_example -> ../versioned_data/mp3d_example_scene_1.1"
fi

cd "${REPO_ROOT}"

# ObjectNav episodes store scene_id as "mp3d/<scene>/<scene>.glb", resolved
# against habitat.dataset.scenes_dir, so something must answer to the name
# "mp3d". That symlink goes in a scene root of its own rather than in
# data/scene_datasets, because the full MP3D download writes to
# data/scene_datasets/mp3d/ — it would follow the symlink and dump 90 scenes into
# the example directory. This way the two can coexist: the example config points
# at scene_datasets_example, a full download at scene_datasets, neither touching
# the other.
mkdir -p data/scene_datasets_example
ln -sfn ../versioned_data/mp3d_example_scene_1.1 data/scene_datasets_example/mp3d
echo "linked data/scene_datasets_example/mp3d -> ../versioned_data/mp3d_example_scene_1.1"

if [ -d data/scene_datasets/mp3d ] && [ ! -L data/scene_datasets/mp3d ]; then
    echo "note: full MP3D detected at data/scene_datasets/mp3d — unaffected by the above"
fi

echo
echo "next: conda run -n vln python tests/mp3d_example_env.py"
