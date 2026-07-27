# Docker setup

Reproduces the LongNav-R1 environment in one image. Everything (habitat-sim,
habitat-lab, torch, project deps) lives in a single conda env named `vln` with
**python 3.10.16**.

## Why the source build

The published `habitat-sim==0.3.3` conda package is built against python 3.9, so
`conda install habitat-sim` would pull the env back to 3.9 and break the rest of
the project (`requires-python >= 3.10`). The image therefore builds habitat-sim
from source with:

```
python setup.py install --headless --with-cuda --bullet
```

This is why the base image is `nvidia/cuda:*-devel-*` rather than `runtime`:
`--with-cuda` needs `nvcc` and the CUDA headers at build time.

## Prerequisites

- NVIDIA driver on the host + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Docker Compose v2

## Build

```bash
export UID GID          # so files written to mounts are owned by you
docker compose build
```

Expect 30–50 minutes; the habitat-sim compile dominates. Useful build args:

| Arg | Default | Notes |
| --- | --- | --- |
| `CUDA_IMAGE` | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` | Match your driver if it's older than 12.8 |
| `PYTHON_VERSION` | `3.10.16` | |
| `HABITAT_VERSION` | `v0.3.3` | Tag used for both habitat-sim and habitat-lab |
| `CMAKE_BUILD_PARALLEL_LEVEL` | `8` | Lower it if the build OOMs |
| `INSTALL_FLASH_ATTN` | `false` | Slow; set `true` for training |

## Run

```bash
docker compose run --rm longnav                              # shell, vln active
docker compose run --rm longnav python docker/verify_install.py
docker compose run --rm longnav python tests/eval_smoke.py
```

## Mounts

| Container path | Source | Purpose |
| --- | --- | --- |
| `/workspace` | repo root | live-edit code; no rebuild needed |
| `/workspace/data` | `$LONGNAV_DATA` (default `./data`) | scene/episode datasets |
| `/home/longnav/.cache` | named volume `longnav-cache` | HF + torch checkpoint cache |

Set `LONGNAV_DATA=/mnt/big-disk/habitat-data` in a `.env` file next to
`docker-compose.yml` when the datasets live outside the repo.

The entrypoint activates `vln` and, on first start, runs
`pip install --no-dependencies -e .` for the repo (and `verl/` if present),
since editable installs of a bind mount can't be baked into the image. Set
`LONGNAV_SKIP_INSTALL=1` to suppress that.

`shm_size` is set to 32GB because Ray's object store and the habitat workers
exhaust the 64MB default immediately; tune it alongside `resources.osm_gb`.

## Known sharp edges

- **numpy is pinned `<2`** image-wide. habitat-sim's magnum bindings are
  compiled against the numpy C API present at build time; letting a later
  `pip install` upgrade numpy silently breaks `import habitat_sim`.
- **`NVIDIA_DRIVER_CAPABILITIES` must include `graphics`**, not just `compute`.
  Headless EGL rendering fails with an opaque context error otherwise.
- The image installs `opencv-python-headless` instead of `opencv-python` to
  avoid pulling X11 runtime deps into a headless container.
