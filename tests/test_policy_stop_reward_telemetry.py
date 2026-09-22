from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np

from longnav.env.objectnav_continuous import ContinuousObjectNavEnvActor


def test_policy_stop_emits_reward_telemetry_columns():
    actor = object.__new__(ContinuousObjectNavEnvActor)
    actor._episode = SimpleNamespace(
        uid="scene:0#1", scene_id="scene", object_category="chair"
    )
    actor._prev_geodesic = 2.0
    actor._min_geodesic = 1.5
    actor._start_geodesic = 5.0
    actor._path_at_min = 3.0
    actor._task = SimpleNamespace(
        path_tracker=SimpleNamespace(length=3.0),
        evaluate=lambda: {"oracle_success": 1.0, "oracle_spl": 0.8},
    )
    actor.success_distance = 1.0
    actor.policy_stop_correct_reward = 3.0
    actor.policy_stop_false_penalty = 3.0
    actor.success_reward = 0.0
    actor.first_reach_reward = 1.0
    actor._max_fall_run = 0
    actor._blind_run = 0
    actor._steps = 1
    actor.dt = 0.04
    actor._exploration_reward_total = 0.25
    actor._cache = {"info": [], "reward": []}
    actor._episodes = None
    actor._render = lambda: np.zeros((1, 1, 3), dtype=np.uint8)
    actor._pos_rots = lambda: []

    state = actor.policy_stop()

    info = state[1]["info"]
    assert info["oracle_reached"] == 1.0
    assert info["oracle_spl"] == 0.8
    assert info["exploration_reward_total"] == 0.25
    assert info["success_continue_penalty"] == 0.0
    assert info["first_reach_reward"] == 0.0
    assert info["timeout_penalty"] == 0.0
    assert info["executed_path_delta_m"] == 0.0
    assert info["path_length_reward_penalty"] == 0.0


def test_first_reach_bonus_is_one_time_and_never_terminates():
    actor = object.__new__(ContinuousObjectNavEnvActor)
    actor.success_distance = 1.0
    actor.first_reach_reward = 1.0
    actor._ever_reached_success = False
    actor._first_reach_reward_awarded = False

    assert actor._record_first_reach(0.9) == 1.0
    assert actor._ever_reached_success is True
    assert actor._first_reach_reward_awarded is True
    assert actor._record_first_reach(0.8) == 0.0


def test_episode_summary_preserves_terminal_stop_and_timeout(tmp_path):
    actor = object.__new__(ContinuousObjectNavEnvActor)
    actor.logging_output_dir = str(tmp_path)
    actor.minimal_logging = True
    actor.logger_actor = None
    actor._episode = SimpleNamespace(object_category="chair")
    for reason, truncated in (("policy_stop", False), ("max_steps", True)):
        actor._cache = {
            "info": [{"episode_label": reason, "scene_id": "scene",
                      "termination_reason": reason, "truncated": truncated}],
            "reward": [0.0],
        }
        path = actor.flush_logs_to_disk()
        summary = json.loads(Path(path).read_text())
        assert summary["termination_reason"] == reason
        assert summary["truncated"] == int(truncated)
