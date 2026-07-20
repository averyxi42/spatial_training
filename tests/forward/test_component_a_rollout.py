"""Component A: run_rollout_cycle -- real production YAML config (see
tests/forward/scenarios/*.yaml), real Ray-actor RLActor, real forward
passes, with the sim backend swapped to the scripted ReplayEnvActor (see
_bootstrap.py) so the episode itself is reproducible run to run.

Parametrized over every scenario in tests/forward/scenarios/ -- one test
instance per production config combination, no hardcoded single case.
Adding coverage for a new config combination means dropping a new yaml file
in that directory, not editing this file.

Input: production YAML config values only (no upstream stored fixture --
this is the first stage of the chain).
Output: the full traj_batch (TensorDict) and model_inputs list
run_rollout_cycle actually returns, stored via torch.save (exact, not
summarized) in tests/forward/fixtures/, one pair of files per scenario.
This is Component B's input.

DISCOVERED AND FIXED during this test's development: the discrete action
sampled at each step (`np.random.choice`, seeded via `np.random.seed(os.getpid())`
in RolloutWorker.__init__) is genuinely random per process. Its *text*
("**forward**" etc.) gets folded back into every subsequent turn's prompt
(substitute_convo_template), so a different sampled action at turn 1 means a
completely different turn>=2 prompt -- and therefore wildly different
turn>=2 model outputs (measured: std up to ~1.7 across fresh processes on
an otherwise identical scripted episode). This is not a bug -- it's normal
on-policy rollout behavior -- but it makes raw trajectories fundamentally
non-reproducible. Fixed via `rollout.use_oracle_action=true` (new
RolloutConfig field, rollout_core.py): run_episode now acts on
state_dict['info']['oracle_action'] (which ReplayEnvActor's script
provides deterministically) instead of the sampled action, while the real
model still computes real logits/probs/logprobs. Verified empirically:
zero measured variance (std=0.0) across 3 fresh-process runs for
rollout_probs/old_logprobs/actions/old_log_prob after this fix, vs. the
large variance before it.
"""
import os

import pytest
import torch

from _bootstrap import bootstrap_with_replay_sim, make_replay_script, teardown
from _compare import diff_traj_batches
from _fixture_io import load_traj_batch, save_model_inputs, save_traj_batch
from scenarios import SCENARIO_IDS, SCENARIOS, Scenario
from longnav.utils.train_loop import run_rollout_cycle

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

# Measured zero (0.0) std across 3 fresh-process runs with the oracle-action
# fix in place -- this tight tolerance is a safety margin above that
# measurement, not a guess, and not loosened to paper over any known jitter.
TOLERANCE = {"atol": 1e-5, "rtol": 1e-4}


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("Component A requires a GPU; none available")


def traj_batch_fixture_name(scenario: Scenario) -> str:
    return f"component_a_traj_batch__{scenario.name}"


def model_inputs_fixture_name(scenario: Scenario) -> str:
    return f"component_a_model_inputs__{scenario.name}"


def _run_component_a(compose_rl_config, scenario: Scenario):
    cfg = compose_rl_config(scenario.overrides)
    script = make_replay_script(n_steps=scenario.n_steps)
    bootstrapper, trainers, sims, wandb_actor, shard_iter, logger = bootstrap_with_replay_sim(cfg, script)
    try:
        traj_batch, model_inputs, values, distances, log_list = run_rollout_cycle(
            sims,
            trainers,
            shard_iter,
            [],
            cfg.training.rl_config.n_rollout,
            cfg.training.rl_config.n_adv,
        )
    finally:
        teardown(trainers, sims, wandb_actor)
    return traj_batch, model_inputs


@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIO_IDS)
def test_component_a_rollout(compose_rl_config, scenario: Scenario):
    _require_gpu()
    traj_batch, model_inputs = _run_component_a(compose_rl_config, scenario)

    print(f"\nComponent A traj_batch keys/shapes/dtypes [{scenario.name}]:")
    for k in sorted(traj_batch.keys()):
        print(f"  {k}: shape={tuple(traj_batch[k].shape)} dtype={traj_batch[k].dtype}")

    if os.environ.get("LONGNAV_UPDATE_FIXTURES") == "1":
        save_traj_batch(traj_batch_fixture_name(scenario), traj_batch)  # full batch: cheap (~14KB), no trimming
        # model_inputs is ~14MB PER trajectory (real per-patch visual
        # embeds at hidden_dim=2048) -- storing all n_rollout would be
        # impractical to commit to git. Component C only ever calls
        # train_rl_step on one trajectory at a time anyway (production
        # dispatches per-trajectory, one call per worker slot), so storing
        # a single representative trajectory's model_inputs exercises the
        # exact same code path with full, uncompressed fidelity for what IS
        # stored -- this is a storage-driven trim of *how many*
        # trajectories are kept, not a summarization of any trajectory's
        # own data.
        save_model_inputs(model_inputs_fixture_name(scenario), model_inputs[:1])
        pytest.skip("fixture (re)captured via LONGNAV_UPDATE_FIXTURES=1; rerun without it to verify")

    expected = load_traj_batch(traj_batch_fixture_name(scenario))
    mismatches = diff_traj_batches(traj_batch, expected, **TOLERANCE)
    assert not mismatches, "\n".join(mismatches)
