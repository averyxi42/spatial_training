"""Component C: RLActor.train_rl_step -- real production yaml (see
tests/forward/scenarios/*.yaml), a real in-process RLActor built the same
way ExpBootstrapper.bootstrap_vlms_rl(training=True) does (see conftest.py's
build_rl_worker), real forward+backward pass, real optimizer.step().

Parametrized over every scenario in tests/forward/scenarios/, matching
Components A/B -- adding a scenario there automatically adds coverage here.

Input: Component A's stored model_inputs fixture for the same scenario
(one representative trajectory's (embeds_inputs_np, embeds_inputs_meta)
tuple) + Component B's stored traj_batch fixture (advantages/returns
already computed), sliced to trajectory 0's valid response tokens exactly
the way train_loop.run_training_epochs does at its call site:
    traj_batch[idx:idx+1, traj_batch["response_mask"][idx].bool()]
Output: the metrics dict train_rl_step returns, stored as plain JSON (already
scalar floats -- no tensor round-tripping needed, so no compression
question arises).

Unlike Components A/B, `train_rl_step` performs a real optimizer.step() --
it mutates the worker's weights and is NOT idempotent (see vlm_worker.py's
generic_train_step: backward + clip_grad_norm_ + optimizer.step() +
scheduler.step() happen unconditionally). So this test always builds a
*fresh* worker via build_rl_worker (cache_key = the scenario name) rather
than reusing the plumbing tests' cached one, and the "repeat and measure"
calibration this file's TOLERANCE is based on was done across separate
fresh pytest-process invocations (fresh CUDA context, fresh weights each
time), not repeated calls on one worker.

Calibration (3 fresh-process runs, discrete_dummy_rpp scenario): loss/pg_loss,
return, train/entropy, train/lr, actor/ppo_kl, actor/pg_clipfrac,
train/ref_kl_divergence and train/rollout_kl_divergence were bit-identical
across all 3 runs. train/grad_norm was NOT bit-identical -- observed values
8.734580039978027, 8.73276424407959, 8.737128257751465 (spread ~0.0044) --
consistent with non-reentrant gradient-checkpointing recomputation feeding
a backward pass whose reduction order/kernel selection isn't pinned (no
torch.use_deterministic_algorithms here). TOLERANCE (atol=1e-4, rtol=1e-3)
is set from that measured spread (~0.0088 allowance at this magnitude), not
a guess and not loosened beyond what was actually observed -- if a future
run's grad_norm exceeds it, that is a real signal, not fixture flakiness to
silently widen away. Per-scenario recalibration may be needed if a new
scenario's magnitudes differ substantially.
"""
import json
import os

import pytest

from _fixture_io import FIXTURES_DIR, load_model_inputs, load_traj_batch
from scenarios import SCENARIO_IDS, SCENARIOS, Scenario
from test_component_a_rollout import model_inputs_fixture_name as component_a_model_inputs_fixture_name
from test_component_b_advantages import traj_batch_fixture_name as component_b_traj_batch_fixture_name

# Measured 0.0 std for the numeric metrics across 3 fresh-process
# calibration runs (see module docstring). Safety margin, not a guess.
TOLERANCE = {"atol": 1e-4, "rtol": 1e-3}

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def metrics_fixture_path(scenario: Scenario) -> str:
    return os.path.join(FIXTURES_DIR, f"component_c_metrics__{scenario.name}.json")


def _run_component_c(build_rl_worker, scenario: Scenario):
    traj_batch = load_traj_batch(component_b_traj_batch_fixture_name(scenario))
    model_inputs = load_model_inputs(component_a_model_inputs_fixture_name(scenario))
    embeds_inputs_np, embeds_inputs_meta = model_inputs[0]

    idx = 0
    traj_slice = traj_batch[idx : idx + 1, traj_batch["response_mask"][idx].bool()]

    worker = build_rl_worker(scenario.overrides, cache_key=scenario.name)
    metrics = worker.train_rl_step(embeds_inputs_np, embeds_inputs_meta, traj_slice)
    return metrics


@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIO_IDS)
def test_component_c_train_step(build_rl_worker, scenario: Scenario):
    metrics = _run_component_c(build_rl_worker, scenario)

    print(f"\nComponent C metrics [{scenario.name}]:")
    for k in sorted(metrics.keys()):
        print(f"  {k}: {metrics[k]}")

    if os.environ.get("LONGNAV_UPDATE_FIXTURES") == "1":
        os.makedirs(FIXTURES_DIR, exist_ok=True)
        with open(metrics_fixture_path(scenario), "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        pytest.skip("fixture (re)captured via LONGNAV_UPDATE_FIXTURES=1; rerun without it to verify")

    with open(metrics_fixture_path(scenario), "r") as f:
        expected = json.load(f)

    mismatches = []
    actual_keys = set(metrics.keys())
    expected_keys = set(expected.keys())
    if actual_keys != expected_keys:
        mismatches.append(f"key set differs: missing={expected_keys - actual_keys} added={actual_keys - expected_keys}")
    else:
        for k in sorted(actual_keys):
            a, e = float(metrics[k]), float(expected[k])
            if abs(a - e) > TOLERANCE["atol"] + TOLERANCE["rtol"] * abs(e):
                mismatches.append(f"{k}: {a} vs {e} (atol={TOLERANCE['atol']}, rtol={TOLERANCE['rtol']})")

    assert not mismatches, "\n".join(mismatches)
