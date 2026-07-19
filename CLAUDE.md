# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LongNav: a Ray-orchestrated RL/eval framework for training VLM-based navigation policies (discrete or continuous action spaces) against Habitat simulator environments, with a dummy env for dependency-free smoke testing. Uses Hydra for config composition and a vendored/submoduled `verl` (RL algorithms, e.g. advantage estimators, PPO core) as a dependency.

## Setup

Two separate conda environments are used because Habitat and the VLM trainer have conflicting dependencies:

```bash
# VLM trainer env (primary dev env)
conda create -n longnav_vlm python=3.10.16 && conda activate longnav_vlm
pip install -e .
pip install flash-attn --no-build-isolation
git submodule update --init --recursive
cd verl && pip install --no-dependencies -e .

# Habitat sim env (separate env named "vln")
# from repo root, with vln active:
pip install --no-dependencies -e .
```

Ray dispatches work to actors running in these different conda envs at runtime (see `resources.vlm_conda_env` / `resources.habitat_conda_env` in config), so both envs must exist on the machine even though a single script invocation only runs from one of them.

## Commands

Pytest suite (run from the `longnav_vlm` env):
```bash
pytest tests/ -m "not gpu"   # CPU-only tier: rollout plumbing + pure RL-math, CI-eligible
pytest tests/forward -m gpu  # GPU tier: real-checkpoint forward-pass tests, local/pre-PR gate only (not yet implemented, see tests/forward/)
```
`pytest tests/rl_math tests/rollout -m "not gpu"` needs no GPU/Ray/checkpoint and is the CI-eligible subset. Tests marked `gpu`/`slow` need a GPU and the real checkpoint and are meant to be run locally before merging changes to the head-polymorphism surface, not in CI.

Standalone smoke-test scripts (run from the `longnav_vlm` env; validate the install; these predate and still coexist with the pytest suite above — they use Ray end-to-end against a real GPU/model, unlike the pytest suite's dummy/mocked coverage):
```bash
python3 tests/eval_smoke.py   # rollout collection only
python3 tests/rl_smoke.py     # one RL training step
```

Dummy-env quickstart (no Habitat dependency required):
```bash
python3 -m longnav.scripts.train_dummy
python3 -m longnav.scripts.train_dummy_continuous
```

Full RL training (Hydra compositional CLI):
```bash
python3 -m longnav.scripts.train_rl +checkpoint=sft +dataset=hm3d_train +experiment=train +resources=octo +training=hapo task.run_name=my_run
```

Eval:
```bash
python -m longnav.scripts.eval +checkpoint=longnav +dataset=hm3d_val +experiment=eval +resources=octo task.run_name=my_eval_run
```

Serving (FastAPI, for sim-to-real):
```bash
python3 -m longnav.serve
# see tools/client.py for an API usage example
```

There is no configured lint/format command. `pytest tests/` is the test runner for `tests/rl_math/`, `tests/rollout/`, and (once implemented) `tests/forward/`; the standalone scripts directly under `tests/` (`rl_smoke.py`, `rl_single.py`, `rl_continuous_single.py`, `eval_smoke.py`) remain separately-runnable Ray/GPU smoke tests, not part of pytest collection.

## Architecture

### Config system (Hydra + dataclasses)

- `src/longnav/config_schema.py` defines every config dataclass (`ResourceConfig`, `VLMConfig`, `RLAlgoConfig`, `HabitatConfig`, `RolloutConfig`, `RunConfig`, and root configs `InferenceConfig` / `RLConfig`). Start here to see what fields exist before editing YAML.
- `src/longnav/config/` holds YAML override groups (`checkpoint/`, `dataset/`, `experiment/`, `resources/`, `training/`) that compose onto the dataclass defaults via Hydra's `+group=name` CLI syntax.
- `src/longnav/conf/register_configs.py` registers the dataclasses with Hydra's `ConfigStore` and defines a custom `read_text` OmegaConf resolver (used e.g. for `${read_text:src/longnav/conf/prompts/...}` interpolations that pull prompt text into config values). Scripts must call `register_configs()` before `@hydra.main` config resolution happens.
- New experiment YAMLs must start with `# @package _global_`.
- Key fields to know when tuning runs: `resources.num_vlms` must not exceed available GPUs and (for training) must divide `n_rollout`; `resources.num_sims` should generally be `num_vlms + 1` or more; `task.run_name` controls wandb resume behavior; `resources.osm_gb` sizes Ray's object store.

### Ray actor orchestration

`src/longnav/utils/factories.py` is the central place where config becomes running infrastructure:
- `ExpBootstrapper` — top-level orchestrator; resolves the Hydra config to a plain dict (for pickling into Ray actors) while keeping the typed dataclass around for direct access. Its `bootstrap_*` methods spin up the Ray cluster, VLM workers, sim workers, and the wandb logger actor.
- `InferenceWorkerFactory` / `RLWorkerFactory` / `SimWorkerFactory` / `WandbFactory` — each wraps a class as a Ray remote actor with the right resource tags (`vlm_resource_tag`/`sim_resource_tag`), GPU fractions, and per-actor conda `runtime_env`. `max_restarts=0` is intentional — actor crashes are meant to surface, not be silently retried.
- VLM and sim actors run in different conda envs (see Setup above); this is why config carries dicts of resolved primitives (not live Python objects) across the Ray actor boundary.

### Rollout / training pipeline

- `src/longnav/utils/vlm_worker.py` — `VLMWorker` (inference: encodes conversation + image into model inputs, samples actions/log-probs) and `VLMTrainingMixin` (adds gradient step, checkpointing, PEFT/LoRA setup, DDP rendezvous). `RolloutWorker`/`RLActor` in `rollout_core.py` combine these via multiple inheritance to get an actor that can both roll out episodes and train.
- `src/longnav/utils/rollout_core.py` — `EpisodeRolloutMixin.run_episode` is the per-episode interaction loop: builds the VLM conversation from `rollout.convo_start_template`/`convo_turn_template` (Python `string.Template` substitution against episode obs), calls the VLM for an action, steps the sim env actor, and packs the trajectory into columnar numpy arrays for zero-copy Ray transfer. `collect_rollouts` fans this out across sim/VLM actor pairs.
- `src/longnav/utils/rl_core.py` — trajectory collation (`collate_trajectories`) and the RL math: advantage estimators (multiple REINFORCE++ time/distance-kernel variants), a `BinnedKernelCritic`, and DAgger-style hybrid splitting. Advantage estimator functions themselves are looked up from `verl.trainer.ppo.core_algos.get_adv_estimator_fn` — `verl` is the source of truth for core PPO/GAE algorithms; this file adds LongNav-specific kernel-based variants on top.
- `src/longnav/scripts/train_rl.py` is the reference orchestration loop: bootstrap cluster → collect rollouts → collate → compute advantages/returns → dispatch per-worker training steps over Ray futures → stream results back into wandb → periodic checkpointing. Read this top-to-bottom to understand a full training cycle; `tests/rl_smoke.py` is a minimal single-cycle version of the same flow using the dummy env.

### Env interface

Any environment usable by `collect_rollouts` must implement the actor interface shown in `src/longnav/env/env_base.py` (`DummyEnvActor` for discrete actions, `DummyContinuousEnvActor` for continuous): `reset()` and `step(action)` both return `(rgb, state_dict)` where `state_dict` has `obs`, `done`, `reward`, `is_exhausted`, `info`; plus `assign_shard(episodes)` and `flush_logs_to_disk()`. RGB is returned as a separate positional value (not nested in the dict) specifically for Ray transfer performance. `src/longnav/env/habitat.py` is the real Habitat-backed implementation of this interface; `src/longnav/env/continuous_env_base.py` and `src/longnav/env/sim.py` provide continuous-action-space variants. `task.env_backend` (`habitat` | `continuous_dummy`) selects which is wired up.

### Directories not part of the package

`verl/` is a git submodule (external RL library, installed separately, imported as `verl.*`). `ovon/` is another vendored external submodule. `misc/`, `data_scripts/`, `dump/`, `outputs/`, `wandb/` are scratch/output/notebook directories, not library code.
