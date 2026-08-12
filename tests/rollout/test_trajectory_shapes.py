"""Plumbing/shape tests for EpisodeRolloutMixin.run_episode against a
scripted ReplayEnvActor and a stubbed (non-model) VLM policy.

Scope, per the regression-test plan: prove the *shape* of what run_episode
produces is correct and stable -- trajectory dict keys per head type, dtypes,
stop-guard behavior, episode termination, value-head gating -- not model
correctness (that's the GPU forward-pass tier, tests/forward/, not yet
implemented).
"""
import numpy as np
import pytest
import ray

from longnav.env.replay import ReplayEnvActor

from _stub_vlm import MINIMAL_ROLLOUT_CONFIG, StubEpisodeWorker


def _make_rgb():
    return np.zeros((4, 4, 3), dtype=np.uint8)


def _make_script(n_steps: int, done_at: int):
    """n_steps entries (including the reset/index-0 entry). `done_at` is the
    step index (1-based, i.e. the step() call count) at which `done` flips."""
    script = []
    for i in range(n_steps):
        script.append(
            {
                "rgb": _make_rgb(),
                "obs": {"instr_or_goal": "reach the goal"},
                "reward": 0.1 * i,
                "done": i == done_at,
                "info": {},
            }
        )
    return script


def _replay_handle(script):
    ReplayActor = ray.remote(ReplayEnvActor)
    return ReplayActor.remote(script=script)


def test_discrete_trajectory_shape(ray_session):
    script = _make_script(n_steps=4, done_at=2)
    env_handle = _replay_handle(script)
    initial_state_ref = ray.get(env_handle.reset.remote())

    # One-hot probs: step 1 -> forward (idx 1), step 2 -> stop (idx 0).
    # Deliberately float64 (numpy's default) -- proves rollout_probs is NOT
    # cast down to float32, unlike rewards (see the dtype assertion below).
    probs_sequence = [
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    ]
    worker = StubEpisodeWorker(policy_head_type="discrete", action_probs_sequence=probs_sequence)

    is_exhausted, final_info, trajectory = worker.run_episode(
        env_handle, initial_state_ref, collect_trajectory=True, compute_value=False
    )

    # Episode ends when the *scripted* done flag fires (step 2 here) --
    # exactly 2 steps recorded, regardless of max_steps=16.
    assert trajectory["actions"].shape == (2,)
    assert trajectory["actions"].tolist() == [1, 0]
    assert "actions_continuous" not in trajectory

    assert trajectory["dones"].tolist() == [False, True]
    # rewards are explicitly cast to float32 by _pack_trajectory.
    assert trajectory["rewards"].dtype == np.float32

    # Known gap (pinned, not fixed here): _pack_trajectory only special-cases
    # a "probs" key, but run_episode only ever writes "rollout_probs" -- so
    # the float32-cast branch is dead code, and rollout_probs keeps whatever
    # dtype np.array() infers (float64 here), unlike rewards.
    assert "probs" not in trajectory
    assert trajectory["rollout_probs"].dtype != np.float32

    assert final_info["steps"] == 2
    assert final_info["instr_or_goal"] == "reach the goal"
    assert is_exhausted is False  # ReplayEnvActor.is_exhausted() is a no-op, always False


def test_continuous_trajectory_shape(ray_session):
    script = _make_script(n_steps=4, done_at=2)
    env_handle = _replay_handle(script)
    initial_state_ref = ray.get(env_handle.reset.remote())

    continuous_sequence = [
        np.array([0.1, -0.2], dtype=np.float32),
        np.array([0.3, 0.4], dtype=np.float32),
    ]
    worker = StubEpisodeWorker(policy_head_type="continuous", continuous_action_sequence=continuous_sequence)

    _, _, trajectory = worker.run_episode(
        env_handle, initial_state_ref, collect_trajectory=True, compute_value=False
    )

    assert trajectory["actions_continuous"].shape == (2, 2)
    np.testing.assert_allclose(trajectory["actions_continuous"], np.array(continuous_sequence))
    assert "actions" not in trajectory
    assert "rollout_probs" not in trajectory


def test_stop_prob_threshold_guard(ray_session, monkeypatch):
    """When the raw sampled action is `stop` but its probability is below
    stop_prob_threshold, run_episode must resample away from `stop`."""
    script = _make_script(n_steps=3, done_at=2)
    env_handle = _replay_handle(script)
    initial_state_ref = ray.get(env_handle.reset.remote())

    rollout_config = dict(MINIMAL_ROLLOUT_CONFIG)
    rollout_config["stop_prob_threshold"] = 0.5

    # Low-but-nonzero stop probability -- below threshold, so the guard must fire.
    probs_sequence = [
        np.array([0.2, 0.8, 0.0, 0.0], dtype=np.float32),
        np.array([0.2, 0.8, 0.0, 0.0], dtype=np.float32),
    ]
    worker = StubEpisodeWorker(
        policy_head_type="discrete",
        action_probs_sequence=probs_sequence,
        rollout_config=rollout_config,
    )

    # Force the "raw" sample to land on stop (index 0) deterministically, both
    # for the initial np.random.choice call and the guard's resample call --
    # the resample call excludes index 0 from its own local array, so
    # returning 0 there means "first non-stop action" (global index 1).
    monkeypatch.setattr(np.random, "choice", lambda *args, **kwargs: 0)

    _, _, trajectory = worker.run_episode(
        env_handle, initial_state_ref, collect_trajectory=True, compute_value=False
    )

    assert (trajectory["actions"] != 0).all(), "stop-prob guard should have forced a non-stop action"


@pytest.mark.parametrize("compute_value", [True, False])
def test_value_head_populated_iff_enabled(ray_session, compute_value):
    script = _make_script(n_steps=3, done_at=1)
    env_handle = _replay_handle(script)
    initial_state_ref = ray.get(env_handle.reset.remote())

    probs_sequence = [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)]
    worker = StubEpisodeWorker(policy_head_type="discrete", action_probs_sequence=probs_sequence)

    _, _, trajectory = worker.run_episode(
        env_handle, initial_state_ref, collect_trajectory=True, compute_value=compute_value
    )

    if compute_value:
        assert "values" in trajectory
        assert trajectory["values"].shape[0] == trajectory["actions"].shape[0]
    else:
        assert "values" not in trajectory
