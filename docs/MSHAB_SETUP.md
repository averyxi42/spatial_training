# ManiSkill-HAB environment — full setup instructions

Branch `mshab`. Everything here was built and verified 2026-08-30/31 on purrgil
(8×A100, driver 550.163, CUDA 12.8 container). Design notes, measurements and the
debugging history live in `docs/MSHAB_INTEGRATION.md`; this file is the do-it-again
recipe.

## 0. Prerequisites

- The container needs a working Vulkan loader + NVIDIA ICD (`vulkaninfo --summary`
  with `DISPLAY` **unset** must list the GPUs). `NVIDIA_DRIVER_CAPABILITIES` must
  include `graphics` (or `all`). The repo's own `docker/Dockerfile` does NOT install
  Vulkan — add `libvulkan1 vulkan-tools` to its `system` stage if rebuilding the image.
- Disk: the conda env is ~6 GB, assets ~2.5 GB. On purrgil the root overlay is nearly
  full — put both on NFS as below.

## 1. Conda env (`mshab`, the third env)

Neither existing env can host SAPIEN (`vln` carries habitat-sim's numpy ABI; the
trainer env must stay clean), so ManiSkill gets its own env. Torch and ray versions
deliberately match the trainer env so Ray object pickles and the cluster agree.

```bash
bash setup/install_mshab_env.sh
```

Which does, idempotently:
- `conda create --prefix /Projects/envs/mshab python=3.10.16`
- `torch==2.8.0+cu128`, `torchvision==0.23.0`, `ray[default]==2.53.0`, `numpy==1.26.4`
- clones `third_party/ManiSkill` (**branch `mshab`** — the MS-HAB fork, sapien==3.0.0b1,
  gymnasium==0.29.1; PyPI mani_skill is NOT what MS-HAB was validated on) and
  `third_party/mshab`, both `pip install -e`
- `pip install --no-dependencies -e .` (this repo) + `hydra-core regex`
- symlink `/workspace/conda/envs/mshab -> /Projects/envs/mshab` so Ray's
  `runtime_env={"conda": "mshab"}` resolves by name
- post-fix: `pip install "setuptools<81"` (sapien 3.0.0b1 imports `pkg_resources`)

## 2. Assets

```bash
export MS_ASSET_DIR=/Projects/spatial_training_mshab/data_mshab   # NEVER ~/.maniskill (root disk)
for d in ycb ReplicaCAD ReplicaCADRearrange; do
  /Projects/envs/mshab/bin/python -m mani_skill.utils.download_asset -y "$d"
done
```

~2.5 GB total; lands under `$MS_ASSET_DIR/data/...`. The rearrange bundle carries the
task plans + spawn data for every task/subtask/split. The 490 GB HF demonstration
datasets are separate and not needed to run the env (see MSHAB_INTEGRATION.md §7 for a
sampled copy under `data_mshab/demo_samples/`).

## 3. Smoke tests

```bash
export MS_ASSET_DIR=/Projects/spatial_training_mshab/data_mshab \
       PYTHONPATH=/Projects/spatial_training_mshab/src CUDA_VISIBLE_DEVICES=<free gpu>
env -u DISPLAY /Projects/envs/mshab/bin/python tools/mshab_smoke.py --subtask pick --steps 30
```

`env -u DISPLAY` is mandatory everywhere: a VNC X server on `DISPLAY` makes SAPIEN's
Vulkan init hang forever. First reset builds the scene (~2–4 min); after that ~12 sim
steps/s. The actor also drops `DISPLAY` itself, but the raylet/driver must not carry it.

## 4. The sim actor (`sim=mshab`)

`longnav.env.mshab.MSHabEnvActor` implements the five-method sim contract. Selection is
the Hydra `sim` group (`override /sim: mshab`); Ray-side env is
`resources.sim_conda_env=mshab` (new field; falls back to `habitat_conda_env`).

Key config knobs (`MSHabEnvConfig` in `src/longnav/conf/env_configs.py`):
- `task`/`subtask`/`split` — MS-HAB task plans (`navigate` has geodesic machinery).
- `action_mode`: `joint` (raw 13-d `pd_joint_delta_pos`) or `base_chunk` (the continuous
  ObjectNav `(gap,3)` SE(2) chunk at `dt` per tick; arm tucked and held).
- `holonomic_base: true` swaps Fetch's forward-only base controller for (vx, vy, ω),
  bounds ±2 m/s / ±π rad/s, **`normalize_action=False`** — the bounds-rescaling trap
  cost a day; see MSHAB_INTEGRATION.md §9. The base is three virtual planar joints, so
  holonomic is a controller choice, not a physics change.
- `fov` is **VERTICAL** radians (1.094 ≈ corpus hfov 79° at 4:3), `torso_lift: 0.0`
  puts the head cam at 1.08 m (corpus rig: 1.185 m, 0.1 m behind the axis — residual
  documented, fixable via a sensor pose offset).
- `scene_index`/`max_plans` — one build config per `num_envs=1` actor.
- `goal_categories` + `success_distance` — ObjectNav-style goals over live scene
  instances (floor-map geodesic). For cross-engine work use the matched mode instead.
- `episode_budget` — actor reports exhausted after N episodes (the trivial-shard eval
  path would otherwise loop forever).

## 5. Episode-matched cross-engine protocol

To run SAPIEN on episodes identical to a habitat ReplicaCAD eval:

- `sim.match_scene_instance: <.../configs/scenes/v3_sc1_staging_13.scene_instance.json>`
  restores the staging furniture (builder-exact conversion `Pose(q_90x)*Pose(pos,rot)`;
  MS-HAB's init-config furniture and clutter are parked away). Requires
  `sim.sim_backend: cpu` — static actors cannot be re-posed in GPU sim (renderer stays
  on GPU; the loop is VLM-bound anyway).
- `sim.match_episodes_json: <objnav_replicacad/val/content/<scene>.scene_instance.json.gz>`
  serves each episode's start pose, category, and **view points** in order; success =
  floor-map geodesic to nearest view point ≤ `success_distance` (habitat's VIEW_POINTS
  rule). The episode file is generated by the recipe in MSHAB_INTEGRATION.md §10 /
  `data_mshab/objnav_replicacad/` (ObjectNav-v1 layout; the content shard must be named
  `<scene>.scene_instance.json.gz` to survive the pinned-shard scene filter).
- `sim.video_layout: headcam` records the bare policy view for like-for-like tiles
  (`tools/tile_matched_videos.py` pairs and hstacks them, fps-normalized).

Reference configs: `experiment/mshab_objnav_matched.yaml` (the matched benchmark run),
`mshab_objnav_70s_fixed.yaml` (category-goal mode), `mshab_flow_deploy.yaml` (first
deploy smoke). Habitat-side counterparts: `replicacad_habitat_diag.yaml` (physical
body) and `replicacad_habitat_diag_kin.yaml` (`sim.body: kinematic` — exact chunk
execution on the navmesh, `longnav/env/kinematic_sim.py`). ReplicaCAD-in-habitat needs
the navmesh staged as `configs/scenes/<scene>.scene_instance.navmesh` and a
`default.scene_dataset_config.json` symlink (both already in `data_mshab`).

## 6. Running an eval (isolated Ray, shared machine)

```bash
export PYTHONPATH=/Projects/spatial_training_mshab/src \
       MS_ASSET_DIR=/Projects/spatial_training_mshab/data_mshab CUDA_VISIBLE_DEVICES=<gpu>
env -u DISPLAY /workspace/conda/envs/longnav_vlm/bin/ray start --head --port=6479 \
    --num-cpus=16 --num-gpus=1 --resources='{"env_a": 1, "env_b": 1}' \
    --object-store-memory=8000000000 --temp-dir=/tmp/ray_mshab --disable-usage-stats
env -u DISPLAY /workspace/conda/envs/longnav_vlm/bin/python -m longnav.scripts.eval \
    +experiment=mshab_objnav_matched resources.ray_address=<ip>:6479
```

Ops rules (each learned the hard way — MSHAB_INTEGRATION.md §10):
- `--temp-dir` must be SHORT (AF_UNIX 107-byte socket path limit kills NFS paths).
- One SAPIEN **GPU-physics** process per card; two PhysX-GPU contexts → CUDA 700.
- Tear the head down by PID (`pgrep -f ray_mshab`), never `ray stop` (shared machine).
- Never `pkill -f` a pattern that appears in the same compound command.
- `ffprobe` a video before shipping it; mid-write copies are moov-less stubs.
