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


def test_success_region_observation_does_not_make_a_transition_terminal(ray_session):
    script = _make_script(n_steps=4, done_at=2)
    script[1]["info"] = {"just_reached": True}
    script[2]["info"] = {"just_reached": False}
    env_handle = _replay_handle(script)
    initial_state_ref = ray.get(env_handle.reset.remote())
    worker = StubEpisodeWorker(
        policy_head_type="continuous",
        continuous_action_sequence=[
            np.array([0.1, -0.2], dtype=np.float32),
            np.array([0.3, 0.4], dtype=np.float32),
        ],
    )

    _, _, trajectory = worker.run_episode(
        env_handle, initial_state_ref, collect_trajectory=True, compute_value=False
    )

    assert trajectory["dones"].tolist() == [False, True]


def test_chain_action_context_uses_executed_chunk_not_credited_chain():
    worker = StubEpisodeWorker(
        policy_head_type="continuous",
        continuous_action_sequence=[np.zeros(1, dtype=np.float32)],
    )
    chain = np.arange(12, dtype=np.float32)
    chunk = np.asarray([[0.1, 0.0, 0.2], [0.2, 0.0, 0.3]], dtype=np.float32)

    class _ChainHead:
        @staticmethod
        def sample_chain_np(_hidden):
            return chain, np.asarray([0], dtype=np.int64), -1.0, chunk

    worker.model = type("Model", (), {"action_head": _ChainHead()})()
    sampled = worker._sample_action_for_state(
        {"h": np.zeros(4, dtype=np.float32)}, None, {}, {}
    )

    credited, executed, _, _, action_text = sampled
    np.testing.assert_array_equal(credited, chain)
    np.testing.assert_array_equal(executed, chunk)
    assert action_text == "0.100,0.000,0.200,0.200,0.000,0.300"
    assert "11.000" not in action_text


def test_trajectory_stop_requires_near_zero_translation_and_yaw():
    worker = StubEpisodeWorker(
        policy_head_type="continuous",
        continuous_action_sequence=[np.zeros(2, dtype=np.float32)],
        rollout_config={
            **MINIMAL_ROLLOUT_CONFIG,
            "stop_execution_mode": "trajectory_length",
            "trajectory_stop_threshold_m": 0.21,
            "trajectory_stop_yaw_threshold_rad": 0.01,
        },
    )
    chunk = np.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
    stop, mode, length = worker._policy_stop_decision(chunk, stop_probability=None)
    assert mode == "trajectory_length"
    assert np.isclose(length, 0.2)
    assert stop

    worker.rollout_config["trajectory_stop_threshold_m"] = 0.19
    stop, _, _ = worker._policy_stop_decision(chunk, stop_probability=None)
    assert not stop

    worker.rollout_config["trajectory_stop_threshold_m"] = 0.21
    rotating_chunk = np.asarray([[0.0, 0.0, 0.2]], dtype=np.float32)
    stop, _, length = worker._policy_stop_decision(rotating_chunk, stop_probability=None)
    assert np.isclose(length, 0.0)
    assert not stop


def test_trajectory_stop_respects_initial_no_stop_prefix():
    worker = StubEpisodeWorker(
        policy_head_type="continuous",
        continuous_action_sequence=[np.zeros(2, dtype=np.float32)],
        rollout_config={
            **MINIMAL_ROLLOUT_CONFIG,
            "stop_execution_mode": "trajectory_length",
            "trajectory_stop_threshold_m": 0.21,
            "trajectory_stop_yaw_threshold_rad": 0.01,
            "trajectory_stop_min_steps": 30,
        },
    )
    chunk = np.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
    assert not worker._policy_stop_decision(chunk, None, decision_index=29)[0]
    assert worker._policy_stop_decision(chunk, None, decision_index=30)[0]


def test_sampled_stop_uses_a_seeded_episode_hazard():
    worker = StubEpisodeWorker(
        policy_head_type="continuous",
        continuous_action_sequence=[np.zeros(2, dtype=np.float32)],
        rollout_config={
            **MINIMAL_ROLLOUT_CONFIG,
            "stop_execution_mode": "sampled",
            "stop_sample_temperature": 1.0,
        },
    )
    worker._policy_stop_rng = np.random.default_rng(0)
    stop, mode, _ = worker._policy_stop_decision(np.zeros(2), stop_probability=0.0)
    assert mode == "sampled"
    assert not stop

    worker._policy_stop_rng = np.random.default_rng(0)
    stop, _, _ = worker._policy_stop_decision(np.zeros(2), stop_probability=1.0)
    assert stop


def test_sampled_stop_feedback_reuses_the_executed_action():
    worker = StubEpisodeWorker(
        policy_head_type="continuous",
        continuous_action_sequence=[np.zeros(2, dtype=np.float32)],
        rollout_config={
            **MINIMAL_ROLLOUT_CONFIG,
            "stop_execution_mode": "sampled",
            "stop_sample_temperature": 1.0,
            "stop_shadow_correct_reward": 3.0,
            "stop_shadow_false_penalty": 3.0,
        },
    )
    worker._policy_stop_rng = np.random.default_rng(0)
    stop, _, _ = worker._policy_stop_decision(np.zeros(2), stop_probability=1.0)
    feedback = worker._shadow_stop_decision(
        {"info": {"distance_to_goal": 0.5}}, probability=1.0
    )

    assert stop
    assert feedback["shadow_stop_action"] == 1.0
    assert feedback["shadow_stop_reward"] == 3.0


def test_categorical_stop_records_the_sampled_policy_action():
    worker = StubEpisodeWorker(
        policy_head_type="continuous",
        continuous_action_sequence=[np.zeros(2, dtype=np.float32)],
        rollout_config={
            **MINIMAL_ROLLOUT_CONFIG,
            "stop_execution_mode": "categorical",
        },
    )
    worker._policy_stop_rng = np.random.default_rng(0)
    stop, mode, _ = worker._policy_stop_decision(
        np.zeros(2), stop_probability=0.2
    )
    assert mode == "categorical"
    assert not stop
    assert worker.last_categorical_stop_action is False
    assert np.isclose(worker.last_categorical_stop_logprob, np.log(0.8))


def test_action_path_stop_does_not_create_a_categorical_policy_action():
    worker = StubEpisodeWorker(
        policy_head_type="continuous",
        continuous_action_sequence=[np.zeros(2, dtype=np.float32)],
        rollout_config={
            **MINIMAL_ROLLOUT_CONFIG,
            "stop_execution_mode": "trajectory_length",
            "trajectory_stop_threshold_m": 0.1,
            "trajectory_stop_yaw_threshold_rad": 0.1,
        },
    )
    worker._policy_stop_decision(np.zeros((1, 3)), stop_probability=None)
    assert worker.last_categorical_stop_action is None
    assert np.isnan(worker.last_categorical_stop_logprob)


def test_categorical_stop_consistency_penalizes_only_large_decoded_motion():
    worker = StubEpisodeWorker(
        policy_head_type="continuous",
        continuous_action_sequence=[np.zeros(2, dtype=np.float32)],
        rollout_config={
            **MINIMAL_ROLLOUT_CONFIG,
            "categorical_stop_action_consistency_penalty": 0.25,
            "categorical_stop_action_consistency_translation_m": 0.1,
            "categorical_stop_action_consistency_yaw_rad": 0.1,
        },
    )
    assert worker._categorical_stop_action_consistency_penalty(
        np.array([[0.05, 0.0, 0.05]], dtype=np.float32)
    ) == 0.0
    assert np.isclose(
        worker._categorical_stop_action_consistency_penalty(
            np.array([[0.2, 0.0, 0.0]], dtype=np.float32)
        ),
        0.25,
    )
    assert np.isclose(
        worker._categorical_stop_action_consistency_penalty(
            np.array([[0.0, 0.0, 0.2]], dtype=np.float32)
        ),
        0.25,
    )


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
