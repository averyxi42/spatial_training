# Forward-pass tier (GPU, real checkpoint)

A chain of three component tests that together fingerprint the real
production pipeline (`train_rl.py` / `train_loop.py`) end to end, with real
Hydra-composed production YAML, a real checkpoint, real Ray actors (where
the production code uses them), and real GPU forward/backward passes. No
mocking, no monkeypatching, no piece tested in isolation from its real
neighbors.

## Design

Each component's *input* is either real production YAML config values, or
the *exact stored output* of the component immediately before it in the
chain. Each component's *output* is stored (torch.save / JSON) as
faithfully as storage allows -- trims are only ever storage-driven (e.g.
keeping one representative trajectory's ~14MB `model_inputs` instead of all
16), never a summarization of the data that IS kept. This is what lets the
three tests, run independently, stand in for one continuous run of the
real pipeline: A's real output literally becomes B's real input, and B's
real output becomes C's real input.

```
production YAML ──▶ Component A ──▶ traj_batch, model_inputs (stored)
                     run_rollout_cycle
                     (real Ray actors, real ReplayEnvActor, real
                      forward pass, rollout.use_oracle_action=true)

traj_batch (A) ─────▶ Component B ──▶ traj_batch w/ advantages, returns (stored)
production YAML       compute_advantages_and_returns
                       (pure CPU tensor math, no GPU/Ray)

traj_batch (B), ─────▶ Component C ──▶ metrics dict (stored as JSON)
model_inputs (A)       RLActor.train_rl_step
production YAML        (real forward+backward pass, real optimizer.step())
```

- `test_component_a_rollout.py`
- `test_component_b_advantages.py`
- `test_component_c_train_step.py`

Supporting infra: `_bootstrap.py` (real cluster/actor bring-up with the sim
backend swapped to the scripted, deterministic `ReplayEnvActor`),
`_fixture_io.py` (exact torch.save/load for TensorDict and model_inputs
fixtures), `_compare.py` (full, non-summarizing per-element comparison).

## Scenarios

None of the three test files hardcode a single config combination. Each is
parametrized over every yaml file in `tests/forward/scenarios/`
(`scenarios.py` auto-discovers them at import time). A scenario file
declares the `compose_rl_config` overrides and the replay-episode length:

```yaml
# tests/forward/scenarios/discrete_dummy_rpp.yaml
overrides:
  - "+checkpoint=longnav"
  - "+experiment=discrete_dummy"
  - "+training=rpp"
  - "sim=replay"
  - "resources.vlm_conda_env=null"
  - "resources.habitat_conda_env=null"
  - "task.logger=null"
  - "rollout.use_oracle_action=true"
n_steps: 8
```

To add coverage for another run setup (a different experiment yaml,
training preset, policy head, continuous vs. discrete head, etc.), drop
another yaml file here -- fixture files are auto-named
`{component}__{scenario_name}.{pt,json}`, and all three test files pick up
the new scenario automatically with no code changes. (A continuous-head
scenario needs its own oracle-action mechanism first --
`make_replay_script`'s `oracle_cycle` is a discrete action-space index, not
meaningful for a continuous head -- see `_bootstrap.py`.)

## Why `rollout.use_oracle_action`

Component A's discrete action sampling (`np.random.choice`) is genuinely
random per process (`RolloutWorker.__init__` seeds via `np.random.seed(os.getpid())`),
and the sampled action's *text* is folded back into every subsequent
turn's prompt (`substitute_convo_template`) -- so a different sampled
action at turn 1 produces a completely different turn>=2 prompt and
therefore wildly different turn>=2 model outputs. This is normal on-policy
rollout behavior, not a bug, but it makes raw rollouts fundamentally
non-reproducible run to run. `RolloutConfig.use_oracle_action` (a real,
opt-in production config field -- not a test-only patch) makes
`EpisodeRolloutMixin.run_episode` act on `state_dict['info']['oracle_action']`
(supplied deterministically by `ReplayEnvActor`'s script) instead of the
sampled action, while the model still runs a real forward pass and its
real logits/probs/logprobs are still what gets recorded. It raises a clear
`RuntimeError` if enabled against an env whose `info` doesn't provide
`oracle_action`, since that key is not (yet) a strict env contract.

## Tolerance calibration

Each test's `TOLERANCE` is set from an actual measured spread across
several fresh-process runs, documented in that test file's own docstring
-- not guessed, and not loosened to hide a real discrepancy:

- **Component A** (real forward pass, `torch.inference_mode()`): after the
  oracle-action fix, measured 0.0 std across 3 fresh-process runs for every
  compared field. Tight tolerance (`atol=1e-5, rtol=1e-4`).
- **Component B** (pure CPU tensor math on a frozen input): no real source
  of jitter. Tight tolerance (`atol=1e-6, rtol=1e-6`).
- **Component C** (real forward+backward pass + real `optimizer.step()`):
  most metrics were bit-identical across 3 fresh-process runs, but
  `train/grad_norm` was not (observed spread ~0.0044 at magnitude ~8.7),
  consistent with non-reentrant gradient-checkpointing recomputation and
  unpinned backward-kernel reduction order. Tolerance (`atol=1e-4,
  rtol=1e-3`) is set from that measured spread. `train_rl_step` performs a
  real, unconditional `optimizer.step()`/`scheduler.step()` and is **not**
  idempotent -- each test run builds a fresh worker (a distinct
  `build_rl_worker` cache key) rather than reusing one across calls.

If a future run trips a tolerance, that's a real signal (a behavior change
somewhere in the chain) -- recalibrate deliberately with fresh measurements
and explain why in the docstring, don't silently widen the number.

## Running this tier

```bash
LONGNAV_UPDATE_FIXTURES=1 pytest tests/forward -v -s   # (re)capture fixtures for every scenario, one command
pytest tests/forward -v                                 # verify every scenario against committed fixtures
```

Local/pre-PR gate only -- never wired into CI (needs a GPU + the real
checkpoint download).
