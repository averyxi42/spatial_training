# Docker setup

Reproduces the LongNav-R1 environment. Two conda envs, both python 3.10.16:

| env | contents |
| --- | --- |
| `vln` | habitat-sim 0.3.3 + habitat-lab built from source, plus the project core |
| `longnav` | the project with its `[vlm]` extra — torch, transformers, verl |

**Naming Convention** `src/longnav/config_schema.py` defaults
`habitat_conda_env="vln"` / `vlm_conda_env="longnav"`, and `src/longnav/utils/factories.py`
hands those to Ray as `runtime_env={"conda": ...}`. Ray re-execs each actor inside the named
env, so both must exist under exactly those names, both must have the project installed, and
both must carry the same `ray` version — a mismatch fails at actor startup, not at import.
That is also why the habitat env gets a project install at all: it only needs the core
(`ray` + `hydra-core`), which is precisely what `pyproject.toml` declares outside the
`[vlm]` extra.

## Requirements

An NVIDIA driver plus the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
Docker Compose v2, and ~60 GB of disk for the image.

## Setup

Four commands, once. The container is a long-lived dev box: start it and leave it running.

```bash
bash docker/setup_env.sh                          # writes .env with your uid/gid
docker compose build                              # 40-60 min (the habitat-sim compile)
docker compose up -d                              # start the dev container
docker compose exec longnav bash docker/install.sh   # install into both envs, ~10 min
```

Check it worked:

```bash
docker compose exec longnav python docker/verify_install.py
docker compose exec longnav conda run --no-capture-output -n vln python docker/verify_install.py
```

You also need the datasets — see [../setup/DATASET_SETUP.md](../setup/DATASET_SETUP.md) and
point `LONGNAV_DATA` in `.env` at them if they do not live in `./data`.

## Daily use

```bash
docker compose exec longnav bash          # shell in the longnav env, at /workspace
docker compose exec longnav python tests/eval_smoke.py
docker compose exec longnav python -m longnav.scripts.eval \
    +checkpoint=longnav +dataset=hm3d_val +experiment=eval +resources=octo \
    task.run_name=my_eval_run
```

`/workspace` is your repo bind-mounted live, so edits on the host take effect immediately —
no rebuild, no reinstall, no restart. Everything is installed editable.

Commands run in the `longnav` env by default, interactively or not. You rarely need the
habitat env directly — Ray activates `vln` for the sim actors itself — but when you do:

```bash
docker compose exec longnav conda run --no-capture-output -n vln python -c "import habitat_sim"
docker compose exec -e LONGNAV_ENV=vln longnav bash      # interactive shell in vln
```

### Container lifecycle — what survives

The conda envs live on the `longnav-conda-envs` named volume, so what `install.sh` writes
outlives the container:

| Command | Installed packages | Notes |
| --- | --- | --- |
| `docker compose stop` / `start` | kept | the normal way to pause work |
| `docker compose exec` | kept | never creates a container |
| `docker compose run --rm` | kept | the envs are on a volume, not the writable layer |
| `docker compose down` | kept | volumes survive; just `up -d` again |
| `docker compose down -v` | **destroyed** | removes volumes; re-run `install.sh` |

The first `up -d` seeds the volume from the image (~15 GB copy, a minute or two). That is
also the catch: **the volume then shadows the image's envs.** After rebuilding the image
with a different `HABITAT_VERSION` or `PYTHON_VERSION`, the old envs are still what you
get, and the rebuild looks like it did nothing. Reset with:

```bash
docker compose down -v                          # drops caches too
docker volume rm spatial_training_longnav-conda-envs   # or just this one
```

Re-running `install.sh` after a reset is fast anyway: pip's cache lives at
`/home/longnav/.cache/pip` inside `longnav-cache`, so nothing large is re-downloaded.

## Installing

`docker/install.sh` is the single install path, used for the container and a bare machine
alike. It populates both envs in one run; pip reads `pyproject.toml`, so the script carries no
dependency list of its own.

```bash
docker/install.sh                        # both envs
docker/install.sh --repo /path/to/repo   # if the repo is not the script's parent
INSTALL_FLASH_ATTN=1 docker/install.sh   # add flash-attn to the VLM env (slow)
```

It installs the project editable into both: the bare core into `vln` and the `[vlm]` extra
into `longnav`, plus `verl` (no deps) into `longnav`. Env names default to `vln` / `longnav`
and can be overridden with `HABITAT_ENV_NAME` / `VLM_ENV_NAME`, but they must then match the
config or Ray will look for envs that do not exist. The script is also baked into the image as
`longnav-install`, for a container with no repo mounted yet.

**flash-attn is not built into the image.** Its `setup.py` imports torch, and with
`--no-build-isolation` it uses the env's torch — which does not exist until the `[vlm]` extra
is installed. So it lives in `install.sh`, after that step, behind `INSTALL_FLASH_ATTN=1`.

## Why habitat-sim is built from source

The published `habitat-sim==0.3.3` conda package targets python 3.9, which would drag the env
below the project's `requires-python >= 3.10`. So it is compiled with
`python setup.py install --headless --with-cuda --bullet`, which is why the base image is
`nvidia/cuda:*-devel-*` rather than `runtime` — `--with-cuda` needs nvcc and the CUDA headers.
This is the 40 minutes of the build, and the reason the stages are ordered the way they are.

## Build layout

Three stages, ordered by volatility. Editing anything in `runtime` can never invalidate the
40-minute compile in `habitat`.

```
system   apt, GL/EGL stack, miniforge, git policy         rarely changes
habitat  vln env, habitat-sim source build, habitat-lab   ~40 min
runtime  longnav env, entrypoint, install script          edit freely
```

`docker build --target habitat -t longnav-base:habitat0.3.3 -f docker/Dockerfile .` snapshots
the expensive stage for pushing to a registry so teammates never compile habitat at all.

Two rules the Dockerfile depends on — please keep them when editing:

1. **Declare an `ARG` immediately before the `RUN` that uses it**, never at the top of a
   stage. An `ARG`/`ENV` change invalidates every layer below it; this is exactly how a
   previous revision made a user-config tweak rebuild habitat-sim.
2. **Every `RUN` that writes into `/opt/conda` ends with `chmod -R a+rwX` in the same `RUN`.**
   A trailing `chmod`/`chown` in its own layer forces an overlayfs copy-up of the whole ~15 GB
   conda tree. That permissive mode is also what lets an arbitrary runtime uid pip-install
   into either env, which `install.sh` relies on.

Build args: `CUDA_IMAGE`, `PYTHON_VERSION` (3.10.16), `HABITAT_VERSION` (v0.3.3),
`CMAKE_BUILD_PARALLEL_LEVEL` (lower it if the build OOMs).

## The user model

The image bakes in **no** uid. It is built as root, and `docker-compose.yml` passes
`user: "${LONGNAV_UID}:${LONGNAV_GID}"` at run time, so switching users is a container
restart, never a rebuild. The entrypoint synthesizes `/etc/passwd` and `/etc/group` entries
for whatever uid it is handed (both files are mode 666), because a uid missing from the passwd
database breaks `sudo`, `id -gn`, and anything resolving `$HOME`.

`docker/setup_env.sh` writes those into `.env`, which compose reads automatically.
**Do not replace them with `${UID}`/`${GID}`**: bash does not export `UID` and does not define
`GID` at all (it is a zsh builtin), so compose would see neither and silently pin everyone to
1000:1000 — which looks like it works right up until someone's gid isn't 1000.

With `user:` set there is no root in the container, so use the preinstalled `sudo` for
`apt-get`.

## Mounts

| Container path | Source | Purpose |
| --- | --- | --- |
| `/workspace` | repo root | live-edit code; no rebuild needed |
| `/workspace/data` | `$LONGNAV_DATA` (default `./data`) | scene/episode datasets |
| `/home/longnav/.cache` | named volume `longnav-cache` | HF + torch caches, bash history |

`shm_size` is 16 GB because Ray's object store and the habitat workers exhaust the 64 MB
default immediately; tune it alongside `resources.osm_gb`. `pids_limit: -1` and
`nofile: 65536` are there because Ray's raylet dies under the docker defaults.

## Known sharp edges

- **The two envs are deliberately asymmetric, and that is not a bug.**
  `vln` holds habitat plus only the project core (`ray`, `hydra-core`) and pins numpy at
  `1.26.4`, because habitat-sim's magnum bindings are compiled against that C API and any
  drift silently breaks `import habitat_sim`. `longnav` gets the `[vlm]` extra and is free
  to run numpy 2.x, and has no torch counterpart in `vln` — the sim actors do not need it,
  and duplicating a multi-GB CUDA wheel per env is not free. Arrays crossing between the
  envs go through Ray's serialisation, not a shared ABI, so the numpy versions need not
  match. `docker/verify_install.py` checks by capability rather than env name.
- **CUDA architectures are listed explicitly** (`TORCH_CUDA_ARCH_LIST` /
  `CMAKE_CUDA_ARCHITECTURES`). No GPU is visible during `docker build`, so nothing can be
  autodetected. Trim the list to shorten the build.
- **The conda env's `libstdc++` is symlinked to the system one.** The env ships an older copy
  than Ubuntu 22.04, and magnum/CUDA extensions otherwise die on `GLIBCXX_3.4.30 not found`.
- **`NVIDIA_DRIVER_CAPABILITIES` must include `graphics`**, not just `compute`, or headless
  EGL fails with an opaque context error.
- **The Dockerfile is not a login shell.** Ubuntu's `/etc/profile` resets `PATH` for uid 0,
  which would wipe the conda and CUDA entries, so `SHELL` is `bash -o pipefail -c`.
- `git config --system --add safe.directory '*'` and a `url.insteadOf` rewrite for
  `git@github.com:` are set image-wide: the former because a bind-mounted repo always looks
  foreign-owned inside the container, the latter because `.gitmodules` declares `verl` and
  `ovon` over SSH and a container has no key.
