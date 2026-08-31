# ManiSkill-HAB as a second simulator backend

Branch `mshab`, worktree `/Projects/spatial_training_mshab`, cut from `code_tokenizer`
@ `f7428be` (2026-08-30). Read-only analysis of the existing stack first, then what was
built. Nothing under `/Projects/spatial_training`, `_tok`, `_rtc`, or the `vln` /
`longnav_vlm` conda envs was touched.

## 1. What MS-HAB is, in this project's terms

[MS-HAB](https://arth-shukla.github.io/mshab/) is the Habitat 2.0 Home Assistant
Benchmark re-implemented on SAPIEN / ManiSkill3 with GPU-parallel physics. Three
long-horizon tasks (TidyHouse, PrepareGroceries, SetTable) over ReplicaCAD apartments,
each decomposed into subtasks: `pick`, `place`, `open`, `close`, and a `navigate`
subtask with a **geodesic** distance reward (`third_party/mshab/mshab/envs/navigate.py`).
Robot is Fetch (mobile base + 7-DoF arm + torso/head).

| | Habitat ObjectNav (current) | MS-HAB |
|---|---|---|
| sim | habitat-sim 0.3.3, Bullet, Y-up | SAPIEN 3.0.0b1 (PhysX GPU), **Z-up**, Vulkan renderer |
| action | SE(2) pose chunk `(gap, 3)`, PID-tracked | joint deltas, **13-d** `pd_joint_delta_pos`: arm 7 · gripper 1 (mimic) · body 3 · base 2 (v, ω) (verified from `action_space`) |
| control rate | 25 Hz ticks, 0.4 s decisions | 20 Hz sim steps (`video_fps=20`); no chunking in the baselines |
| obs | 640×480 RGB | Fetch head + hand cameras, **128×128** (ManiSkill default; resizable via `sensor_configs`), RGB/depth/seg; baselines use depth |
| success | geodesic ≤ `success_distance` | per-subtask `evaluate()`: `success`, `fail`, `is_grasped`, `subtask_steps_left` |
| reward | progress − penalties | `normalized_dense` per subtask; `SequentialTask-v0` is sparse |
| episode identity | dataset episode uid | (scene build config, task plan, spawn index) — sampled per reset, pinnable via `reset(options=...)` |
| package source | `objectnav_eval` from `/Projects/habitat_physical_nav` (undeclared) | `mani_skill` **`mshab` branch** + `mshab` (undeclared, vendored under `third_party/`) |

Version pins that matter (`third_party/ManiSkill/setup.py`): `sapien==3.0.0.b1`,
`gymnasium==0.29.1`, `numpy<2`, `pytorch_kinematics==0.7.5`, `mplib==0.1.1`. PyPI
`mani_skill 3.0.1` / `sapien 3.0.3` are **not** what MS-HAB was validated on; the `mshab`
branch is a 3.0.0b18 fork.

## 2. Is the current docker environment sufficient?

Two separate things called "the docker environment":

**(a) The container we are actually running in** — not the repo's `docker/` image. It is
an Isaac-Sim/webvnc-flavoured image (`/isaac-sim`, `/webvnc`, conda at `/workspace/conda`,
CUDA 12.8, driver 550.163, 8×A100-80GB, 256 cores, 2 TB RAM). For SAPIEN it is
**sufficient as-is**:

- Vulkan loader + NVIDIA ICD present (`/etc/vulkan/icd.d/nvidia_icd.json`, api 1.3.277);
  `vulkaninfo` enumerates the A100s headless (must `unset DISPLAY`; `DISPLAY=:19` points at
  the VNC X server and makes vulkaninfo hang). `NVIDIA_DRIVER_CAPABILITIES=all`.
- SAPIEN offscreen GPU render verified: 256×256 colour frame from a fresh scene
  (`tools/`-less one-liner, 2026-08-30). SAPIEN warns it cannot find the ICD in
  `/usr/share/vulkan/icd.d` and synthesises one; that works.
- CUDA 12.8 torch wheels match the driver.
- **Disk is the real constraint**: the root overlay (`/`, which holds `/workspace/conda`)
  is 99 % full with ~30 GB free (`/root/.cache` alone is 201 GB). The new env therefore
  lives on NFS at `/Projects/envs/mshab` (5.8 GB) with a symlink
  `/workspace/conda/envs/mshab -> /Projects/envs/mshab` so Ray's
  `runtime_env={"conda": "mshab"}` resolves by name. Assets go to
  `/Projects/spatial_training_mshab/data_mshab` (`MS_ASSET_DIR`), never `~/.maniskill`.
- Note `nproc` reports 4 inside the cgroup even though `/proc/cpuinfo` has 256 —
  Ray's default CPU count will be small; pass `num_cpus` explicitly when starting Ray.

**(b) The repo's own `docker/Dockerfile`** (reproduction image, identical in `_tok`) —
**not sufficient**. It installs EGL/OSMesa/GLFW but **no Vulkan loader or ICD**, and
builds exactly two conda envs. To support this branch it needs: `libvulkan1
vulkan-tools` in the `system` stage, the NVIDIA Vulkan ICD (comes with the host driver
when `NVIDIA_DRIVER_CAPABILITIES` includes `graphics`/`all`), and a fourth stage creating
the `mshab` env (`setup/install_mshab_env.sh` is the recipe). The habitat stage's numpy
1.26.4 ABI pin is scoped to `vln` and unaffected.

## 3. What the new conda env contains and why a third env

`vln` cannot host SAPIEN: it carries habitat-sim's compiled numpy ABI, torch 2.7.0+cu128,
vllm 0.9.2, transformers 4.51 — adding `sapien==3.0.0b1` + `gymnasium==0.29.1` +
`pytorch_kinematics` into that would be a resolver fight with no upside, and a broken
`vln` would take down every Habitat run. `longnav_vlm` must stay the trainer env. So:

```
mshab  (python 3.10.16, /Projects/envs/mshab)
  torch 2.8.0+cu128, torchvision 0.23.0     # == longnav_vlm, so Ray object pickles agree
  ray[default] 2.53.0                       # == both existing envs (cluster requirement)
  numpy 1.26.4                              # mshab branch needs <2; keep parity anyway
  sapien 3.0.0b1, mani_skill 3.0.0b18 (mshab branch, editable), mshab (editable)
  gymnasium 0.29.1, setuptools<81           # sapien imports pkg_resources
  longnav (editable, --no-dependencies) + hydra-core + regex
```

Recipe: `setup/install_mshab_env.sh` (idempotent; log in `setup/install_mshab_env.log`).
Assets (`ycb`, `ReplicaCAD`, `ReplicaCADRearrange`): `python -m mani_skill.utils.download_asset`
with `MS_ASSET_DIR` exported — the rearrange zip contains the task plans and spawn data
for every task/subtask/split. The 490 GB demonstration datasets on HF are **not** needed
for RL/eval and were not downloaded.

## 4. How the existing stack accommodates a new sim

The sim boundary was already generalised once (Habitat discrete → continuous ObjectNav;
`docs/LATENT_RL_ENV.md` §C is the playbook). Findings from the read-only sweep:

- **Contract** = five duck-typed methods, no base class: `reset()`,
  `step(action, supplementary_logs=None)`, `assign_shard(list|None)`,
  `flush_logs_to_disk()`, `is_exhausted()`; optional `list_episode_uids()`,
  `set_log_prefix()`. Return `(rgb uint8 HxWx3, {obs, reward, done, is_exhausted, info})`.
- **Selection** is the Hydra `sim` config group (`conf/env_configs.py`), *not* an
  `env_backend` field (CLAUDE.md is stale on this). Experiment yamls must use
  `defaults: - override /sim: <name>`.
- **Only `obs["instr_or_goal"]` is consumed** by the rollout loop; every other obs key is
  exposed to prompt templates as `$key`.
- **Actions are never parsed from text.** Continuous heads emit a vector that goes
  straight to `step()` (`rollout_core.py` ~L226–289); `head.decode_action` is the
  VLM-side actuator seam.
- **`info` must have the same key set on every step** (`_pack_trajectory` takes columns
  from step 0). Values must be collatable scalars/arrays.
- `advantage_estimator: distance_kernel` requires `info["distance_to_goal"]`; the other
  estimators need nothing beyond `reward`/`done`.
- **The one framework change needed**: sim actors' conda env was a single global
  `resources.habitat_conda_env`. Added `resources.sim_conda_env` (falls back to
  `habitat_conda_env`) in `config_schema.py` + `factories.SimWorkerFactory`.
- Habitat leaks outside `env/habitat.py` are cosmetic (log-quiet env vars, uid format
  comments) except `bev_utils.get_cv_to_habitat_correction` (Y-up assumption) which only
  matters for BEV/pose-injection modes.
- `sim_gpu_fraction=0.14` is sized for habitat-sim; SAPIEN GPU sim + Vulkan needs more
  (the smoke yaml sets 0.3; measure).

## 5. What was built on this branch

| file | role |
|---|---|
| `src/longnav/env/mshab.py` | `MSHabEnvActor`: wraps `<Subtask>SubtaskTrain-v0` / `SequentialTask-v0`, `num_envs=1`, one env step = `gap` sim steps, RGB from `fetch_head`, uniform info keys, exhausted sentinel |
| `src/longnav/conf/env_configs.py` | `MSHabEnvConfig`, registered as `sim=mshab` |
| `src/longnav/config_schema.py`, `utils/factories.py` | `resources.sim_conda_env` |
| `src/longnav/config/experiment/mshab_pick_smoke.yaml` | Gaussian head, 14-d action, pick/TidyHouse/train |
| `setup/install_mshab_env.sh` | env recipe |
| `tools/mshab_smoke.py` | Ray-free contract check with random actions |
| `third_party/{ManiSkill,mshab}`, `data_mshab/` | vendored (gitignored) |

## 6. Open design questions (decide before any training run)

1. **Action representation for the VLM.** 14 joint deltas at 20 Hz is a different beast
   from 3-d SE(2) chunks at 2.5 Hz. Options: (a) Gaussian head over 14-d, single step —
   the smoke config; (b) reuse the flow head with a `(T, 14)` chunk and `gap`, retraining
   the chunk tokenizer; (c) `pd_ee_delta_pose` (7-d EE + base) to shrink the space. MS-HAB's
   own baselines are (a)-style PPO/SAC on depth, ~4000 sps with 100s of parallel envs —
   our one-env-per-actor Ray layout will be far slower; see 3.
2. **Observation.** 128×128 head camera is what the benchmark ships; the actor requests
   256×256 by default. The VLM was SFT'd on 640×480 Habitat frames; there is no MS-HAB SFT
   corpus in this project, so RL from the nav checkpoint is off-distribution from step 0.
3. **Throughput.** One SAPIEN scene per Ray actor forfeits ManiSkill's GPU batching. A
   `num_envs=N` actor that serves N episodes to N VLM calls would need `collect_rollouts`
   to understand multi-episode actors — a real change in `rollout_core.py`.
4. **Episode identity / eval pinning.** MS-HAB samples (plan, spawn) per reset;
   `reset(options={"spawn_selection_idxs": [...], "build_config_idxs": [...]})` pins them.
   `assign_shard` currently only budgets episode counts; a uid scheme is TODO.
5. **Frames.** SAPIEN is Z-up; do not route MS-HAB through `bev_utils`/pose injection
   without a new correction matrix.

## 7. Verification run (2026-08-30): SFT flow policy driving Fetch in ReplicaCAD

`src/longnav/config/experiment/mshab_flow_deploy.yaml` -- the continuous ObjectNav
flow policy (`run_cotrain_v3_rtc_ft9000_geo/checkpoint-12000`, pure ODE, `dt=0.04`,
`gap=10`) on `NavigateSubtaskTrain-v0`, TidyHouse scene 0, goal text "chair", arm tucked,
head camera 640x480 at 1.4 rad FOV, lateral motion dropped by the non-holonomic base.
Run through the unmodified `longnav.scripts.eval` Ray path on an isolated Ray head pinned
to GPU 7 (`ray start --head --port 6479 --num-gpus 1 --resources '{"env_a":1,"env_b":1}'`).

Result: 3 episodes x 60 policy steps (480 sim steps = 24 s each), base path length
6.3 / 7.8 / 7.9 m, no falls or force failures; the policy produces coherent forward
chunks (~0.7 m per 0.4 s) and turns at walls/furniture. Videos (20 fps, 768x512):
`dump/mshab_deploy/mshab_flow_deploy/rollout/videos/tidy_house_navigate_train_{0,1,2}.mp4`
-- left: MS-HAB's torso-mounted third-person `render_camera` with the executed trail
(white) and the current chunk projected in 3-D (orange path, cyan heading ticks, yellow
posts); right: the policy's head-camera observation and a body-frame top-down of the
chunk. The green sphere is MS-HAB's navigate-goal marker.

Gotchas found on the way (all fixed in this tree):
* `DISPLAY=:19` (the container's VNC X server) inherited by Ray workers makes SAPIEN's
  Vulkan init block forever -- the actor now pops `DISPLAY`; start any Ray head with
  `env -u DISPLAY`. `vulkaninfo` hangs the same way.
* `flow_sde_policy.sample_chain_np` used a CPU generator against CUDA `randperm` when
  `sde_seed` is set on this branch -- re-homed on the parameters' device.
* MS-HAB's root-joint qpos is relative to the spawn root; world base pose must come
  from `agent.base_link.pose`.
* One-scene constraint: `num_envs=1` needs a plan pool covering exactly one build
  config (`scene_index`, `max_plans`).
* Scene build + navmesh takes 2-4 min per actor (longer under the 1-thread caps);
  `sim_gpu_fraction 0.3`, ~3 GB VRAM for the sim, ~15 GB for the 2B VLM.
* The base tracker's yaw gain overshoots ~50 % on pure-rotation chunks; tune
  `k_w`/feedforward before using MS-HAB numbers for anything quantitative.

## 8. Embodiment vs renderer split (2026-08-30, kinematic body)

`longnav/env/kinematic_sim.py` + `sim.body: kinematic`: cylinder agent, chunks executed
EXACTLY on the navmesh (`pathfinder.try_step`, the habitat-lab `VelocityAction`
mechanism), no Bullet, no PID; camera mount parsed from the physical rig's URDF
(1.185 m, 0.1 m back, 79 deg hfov). Same 24 generated ReplicaCAD ObjectNav episodes
(`data_mshab/objnav_replicacad/val`, scene v3_sc1_staging_13 = MS-HAB eval scene),
same checkpoint (cotrain-v3 rtc_ft9000_geo ck12000, ODE), stock eval stack.

| body / renderer | SR | oSPL | notes |
|---|---|---|---|
| SAPIEN, Fetch PD + hand tracker (holonomic v2) | 2/12 | 0.08 | 1 m point-goal, 40 s budget |
| habitat, physical robot + PID | 21/24 | 0.60 | view-point rings, 70 s |
| habitat, kinematic exact | **24/24** | **0.66** | view-point rings, 70 s |

Reading: category/content grounding transfers to ReplicaCAD; the physical->kinematic
delta (+3 successes, all tv_monitor, +0.06 oSPL) is the PID/embodiment term inside
habitat -- real but small. The dominant term of the MS-HAB collapse is the SAPIEN
side (renderer and/or the Fetch-PD execution); protocol differences (success region,
budget) inflate the habitat-vs-SAPIEN margin and a matched-protocol rerun is still
owed. Episode gotcha: with a pinned shard the loader filters shards by uid stem, so
the content file must be named `<scene>.scene_instance.json.gz`.

## 9. The SAPIEN drive defect (2026-08-31): action normalization

Root cause of the "incompetent movement" in every SAPIEN run up to here: ManiSkill
velocity controllers default to `normalize_action=True` -- the action is treated as
[-1, 1] and rescaled onto the controller bounds. Stock Fetch base bounds are +-1 m/s,
so physical units worked by coincidence; the holonomic patch widened bounds to +-2,
silently DOUBLING every command the chunk tracker sent (measured: cmd 1.0 -> steady
1.62-1.99 m/s; stock: 0.94-0.99). Feedback corrections were 2x too hot per 50 ms tick
-> chunk-scale oscillation/saturation, while long same-direction stretches still built
speed. Fixed with `normalize_action=False` in `_patch_fetch_holonomic`.

Probe methodology gotcha that misled the first measurements: chunk-tracking tests run
inside the live apartment get contaminated by furniture/wall contact (asymmetric
per-direction execution is the tell). Measure drive health with constant-velocity
commands from fresh spawns in BOTH directions (`dump/probe_base_drive2.py`); the
clean signature was: accel ~5 m/s^2 (tau ~0.15 s, comparable to the corpus PID's
a_max 8), symmetric, no force starvation -- gain error was the only real defect.
All SAPIEN evals before `mshab_objnav_70s_fixed` ran with the broken gain and are
superseded, per the rerun-the-corrected-condition rule.

## 10. Episode-matched cross-engine protocol (2026-08-31)

`sim.match_scene_instance` + `sim.match_episodes_json` on `sim=mshab` make a SAPIEN run
EPISODE-IDENTICAL to the habitat ReplicaCAD eval: the staging furniture arrangement is
restored from the scene_instance.json (identity planar mapping, habitat (x,y,z) ->
sapien (x,-z,y); MS-HAB's init-config furniture/clutter is parked far away), and each
reset serves the next episode from the habitat-generated ObjectNav file: same start
pose, same category, and the SAME VIEW POINTS as the success surface (geodesic to
nearest view point <= success_distance -- habitat's VIEW_POINTS rule verbatim; the
floor-map geodesic's vert-snap residual is small for navigable view points, unlike
object centers). Remaining differences between engines: renderer, robot body, drive.
Do NOT compare category-mode SAPIEN runs (live init-config goals, floor-map-to-center
metric) against habitat runs -- different rooms, different measure.

Ops rules learned the hard way: one SAPIEN GPU sim per card (two PhysX GPU contexts on
one device -> CUDA 700 crashes); kills and launches NEVER share a tool call/pattern
(pkill self-match killed launchers three times); ffprobe a video before shipping it
(copying mid-write sends a moov-less stub).

### Camera-geometry residual (measured 2026-08-31)

Rendered intrinsics of the matched SAPIEN runs: fx=394.1 -> hfov 78.2 deg vs the corpus
79.0 (vfov 62.7 vs 63.4) -- FOV is matched; the `fov` param on `sensor_configs` is
VERTICAL. The remaining "zoomed" impression in tiles is the mount: the Fetch head cam
sits ~0.25 m further forward and ~0.10 m lower (1.08 m vs 1.185 m) than the tidybot rig
camera (which is 0.1 m BEHIND the base axis). Fixable exactly with a local pose offset
in the fetch_head sensor config; left as-is in `mshab_objnav_matched` (documented
viewpoint shift), pending a decision on a geometry-exact rerun.

### tv_monitor category caveat (2026-08-31)

Probed after "TV never visible" reports: the matched restore places tv_screen EXACTLY
at its staging pose (static, stable over 20 steps) -- no SAPIEN bug. The real issue is
episode design in BOTH engines: the TV is wall-mounted at 1.9 m with floor-level
view-point rings, and end-frames of successful habitat episodes show the robot reaching
the ring while the TV sits at the frame edge at best. tv_monitor successes in this
scene measure layout-prior navigation, not TV recognition. Treat the category as
weakly grounded in v3_sc1_staging_13; regenerate with visibility-checked view points
(render-at-viewpoint + object-pixel test) before drawing category-level conclusions.
Also: FOV measured matched (hfov 78.2 vs 79.0); the only real optical residual is the
camera mount offset (SETUP.md / camera-geometry note above).

## 11. Final matched benchmark (2026-08-31, run mshab_objnav_matched)

Same 24 episodes (starts, categories, VIEW_POINTS success <= 1.0 m, 70 s budget),
same staging furniture, three conditions:

| condition | SR | oSPL | mean path |
|---|---|---|---|
| habitat, physical robot + PID | 21/24 | 0.60 | -- |
| habitat, kinematic exact | 24/24 | 0.66 | 10.5 m |
| SAPIEN matched, fixed drive + tracker | **22/24** | **0.52** | 14.3 m |

SAPIEN per category: chair 6/6 (oSPL 0.77), plant 6/6 (0.43), sofa 6/6 (0.72),
tv_monitor 4/6 (0.16; the two failures are the weakly-grounded category, see the
tv_monitor caveat -- habitat kinematic also scored its lowest oSPL there). Bottom line:
after the drive-gain fix and the matched protocol, the SAPIEN gap is ~2 episodes of SR
and ~0.14 oSPL against the kinematic-habitat upper bound -- residuals: renderer style,
camera mount offset (~0.25 m forward / 0.10 m low), PD-drive tracking vs exact
execution, and the tv_monitor episode design. The original "2/12 collapse" was, in
order of impact: controller action-normalization bug, protocol mismatch (metric,
budget, different furniture/goals), and only lastly any visual transfer gap.
