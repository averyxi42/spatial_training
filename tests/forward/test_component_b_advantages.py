"""Component B: compute_advantages_and_returns -- real production yaml (see
tests/forward/scenarios/*.yaml) for cfg.training.rl_config, real
verl.trainer.ppo.core_algos.get_adv_estimator_fn lookup (exactly how
train_rl.py obtains it), real production function.

Parametrized over every scenario in tests/forward/scenarios/, matching
Component A -- adding a scenario there automatically adds coverage here too.

Input: Component A's stored traj_batch fixture for the same scenario
(frozen -- not regenerated; run_rollout_cycle is not re-invoked here) +
production yaml config values.
Output: the same traj_batch with `advantages`/`returns` (and, for
estimators that produce one, `baseline`) added, stored via torch.save
(exact, not summarized) in tests/forward/fixtures/. This is Component C's
traj_batch input.

No GPU/Ray/model involved -- this is pure CPU tensor math on a frozen
input, so unlike Component A there is no jitter to calibrate against: two
runs given the same input tensors and the same estimator function are
expected to be bit-for-bit identical. Tolerance is kept tight accordingly.
"""
import os

import pytest
from verl.trainer.ppo.core_algos import get_adv_estimator_fn

from _compare import diff_traj_batches
from _fixture_io import load_traj_batch, save_traj_batch
from scenarios import SCENARIO_IDS, SCENARIOS, Scenario
from test_component_a_rollout import traj_batch_fixture_name as component_a_traj_batch_fixture_name
from longnav.utils.train_loop import compute_advantages_and_returns

# Pure CPU tensor math on a frozen input -- no real source of run-to-run
# jitter, unlike Component A's real GPU forward pass. Tight tolerance is a
# safety margin for float accumulation order, not a measured allowance for
# nondeterminism.
TOLERANCE = {"atol": 1e-6, "rtol": 1e-6}


def traj_batch_fixture_name(scenario: Scenario) -> str:
    return f"component_b_traj_batch__{scenario.name}"


def _run_component_b(compose_rl_config, scenario: Scenario):
    cfg = compose_rl_config(scenario.overrides)
    traj_batch = load_traj_batch(component_a_traj_batch_fixture_name(scenario))
    advantage_estimator_fn = get_adv_estimator_fn(cfg.training.rl_config.advantage_estimator)
    traj_batch, global_return_mean = compute_advantages_and_returns(traj_batch, advantage_estimator_fn, cfg)
    return traj_batch, global_return_mean


@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIO_IDS)
def test_component_b_advantages(compose_rl_config, scenario: Scenario):
    traj_batch, global_return_mean = _run_component_b(compose_rl_config, scenario)

    print(f"\nComponent B traj_batch keys/shapes/dtypes [{scenario.name}]:")
    for k in sorted(traj_batch.keys()):
        print(f"  {k}: shape={tuple(traj_batch[k].shape)} dtype={traj_batch[k].dtype}")
    print(f"global_return_mean: {global_return_mean:.6f}")

    if os.environ.get("LONGNAV_UPDATE_FIXTURES") == "1":
        save_traj_batch(traj_batch_fixture_name(scenario), traj_batch)  # full batch: cheap (~14KB), no trimming
        pytest.skip("fixture (re)captured via LONGNAV_UPDATE_FIXTURES=1; rerun without it to verify")

    expected = load_traj_batch(traj_batch_fixture_name(scenario))
    mismatches = diff_traj_batches(traj_batch, expected, **TOLERANCE)
    assert not mismatches, "\n".join(mismatches)
