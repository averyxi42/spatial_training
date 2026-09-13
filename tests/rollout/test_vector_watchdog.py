import threading
import time
import os
import types

import numpy as np
import pytest
import ray
import torch

from longnav.env.navverse import (
    NavVerseHostActor,
    _native_termination_flags,
)
from longnav.utils.rollout_core import (
    EpisodeRolloutMixin,
    RLWorker,
    _align_readouts_to_executed_actions,
    collect_vector_rollouts,
)
from longnav.utils.train_loop import recycle_vector_sims


def test_tidybot_batch_pose_write_reuses_matching_pose_snapshot():
    root_pose = torch.tensor(
        [
            [1.0, 2.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, 0.5, 1.0, 0.0, 0.0, 0.0],
            [5.0, 6.0, 0.6, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    class Robot:
        device = torch.device("cpu")

        def __init__(self):
            self.data = types.SimpleNamespace(root_state_w=root_pose.clone())
            self.write_count = 0

        def write_data_to_sim(self):
            self.write_count += 1


    robot = Robot()
    calls = []

    class Env:
        navmesh_interface = object()
        collision_response = "block_or_slide"

        def __init__(self):
            self.scene = {"robot": robot}

        @staticmethod
        def _robot_root_z_for_navmesh_xy(xy, fallback_z):
            calls.append(("height", tuple(xy), float(fallback_z)))
            return float(fallback_z) + 0.01

        @staticmethod
        def _is_navmesh_point_clear(start):
            calls.append(("clear", tuple(start)))
            return True

        @staticmethod
        def _is_navmesh_pose_valid(start, candidate):
            calls.append(("valid", tuple(start), tuple(candidate)))
            return float(candidate[0]) < 20.0

        @staticmethod
        def _slide_target_on_navmesh(start, candidate):
            calls.append(("slide", tuple(start), tuple(candidate)))
            return (start + candidate) * 0.5

        @staticmethod
        def _write_tidybot_base_root_pose(updated):
            robot.data.root_state_w[:, :7] = updated

        @staticmethod
        def _remember_tidybot_base_pose(updated):
            calls.append(("remember", updated.detach().clone()))


    host = object.__new__(NavVerseHostActor)
    host.vln_sim = types.SimpleNamespace(env=Env())
    host.profile_step_timing = False
    host.host_profile_stats = {}
    transforms = np.asarray(
        [
            [10.0, 11.0, 0.7, 0.1, 0.2, 0.3, 0.9],
            [12.0, 13.0, 0.8, 0.2, 0.3, 0.4, 0.8],
            [20.0, 21.0, 0.9, 0.3, 0.4, 0.5, 0.7],
        ],
        dtype=np.float64,
    )
    current_positions = root_pose[:, :3].numpy().astype(np.float64)

    host._apply_tidybot_transform_batch(transforms, [0, 2], current_positions)

    expected_position_0 = np.asarray([10.0, 11.0, 0.41], dtype=np.float32)
    candidate_2 = np.asarray([20.0, 21.0, 0.61], dtype=np.float32)
    expected_position_2 = (current_positions[2].astype(np.float32) + candidate_2) * 0.5
    expected_position_2[2] += 0.01
    np.testing.assert_array_equal(
        robot.data.root_state_w[0, :3].numpy(), expected_position_0
    )
    np.testing.assert_array_equal(
        robot.data.root_state_w[2, :3].numpy(), expected_position_2
    )
    np.testing.assert_array_equal(
        robot.data.root_state_w[1].numpy(), root_pose[1].numpy()
    )
    np.testing.assert_array_equal(
        robot.data.root_state_w[0, 3:7].numpy(),
        np.asarray([0.9, 0.1, 0.2, 0.3], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        robot.data.root_state_w[2, 3:7].numpy(),
        np.asarray([0.7, 0.3, 0.4, 0.5], dtype=np.float32),
    )
    assert robot.write_count == 1
    assert [result["blocked"] for result in host.vln_sim.env.last_kinematic_result] == [
        False,
        True,
    ]
    assert [call[0] for call in calls] == [
        "height",
        "clear",
        "valid",
        "height",
        "clear",
        "valid",
        "slide",
        "height",
        "remember",
    ]


def test_tidybot_contact_state_uses_planar_force_across_substeps():
    forces = torch.zeros((2, 8, 1, 3), dtype=torch.float32)
    forces[0, 3, 0, 2] = 100000.0
    forces[1, 5, 0, :2] = torch.tensor([3.0, 4.0])
    sensor = types.SimpleNamespace(
        cfg=types.SimpleNamespace(force_threshold=1.0),
        data=types.SimpleNamespace(
            net_forces_w=torch.zeros((2, 1, 3), dtype=torch.float32),
            net_forces_w_history=forces,
        ),
    )
    host = object.__new__(NavVerseHostActor)
    host.n = 2
    host.vln_sim = types.SimpleNamespace(
        env=types.SimpleNamespace(
            scene=types.SimpleNamespace(sensors={"contact_forces": sensor})
        )
    )

    flags, force_max = host._tidybot_contact_state()

    np.testing.assert_array_equal(flags, np.asarray([False, True]))
    np.testing.assert_allclose(force_max, np.asarray([0.0, 5.0]))


def test_watchdog_race_drops_only_unexecuted_policy_readout():
    trajectory = {
        "actions_continuous": np.zeros((173, 1024), dtype=np.float32),
        "termination_reason": np.asarray(
            [None] * 172 + ["wall_timeout_soft_stop"], dtype=object
        ),
    }
    logits = list(range(174))
    values = list(range(174))

    dropped = _align_readouts_to_executed_actions(
        trajectory, logits, values, "continuous"
    )

    assert dropped == 1
    assert logits == list(range(173))
    assert values == list(range(173))


def test_non_watchdog_readout_mismatch_is_rejected():
    trajectory = {
        "actions_continuous": np.zeros((2, 4), dtype=np.float32),
        "termination_reason": np.asarray([None, "time_out"], dtype=object),
    }

    with pytest.raises(RuntimeError, match="without a watchdog stop"):
        _align_readouts_to_executed_actions(
            trajectory, [0, 1, 2], [], "continuous"
        )


@ray.remote(max_restarts=0, max_task_retries=0)
class _WatchdogEnv:
    def __init__(self, reset_delay=0.0):
        self.reset_delay = float(reset_delay)
        self.labels = []
        self.soft_stops = 0

    def assign_shard(self, labels):
        self.labels = list(labels)

    def reset_vector(self):
        time.sleep(self.reset_delay)
        rgb = np.zeros((len(self.labels), 2, 2, 3), dtype=np.uint8)
        states = [
            {
                "obs": {"instr_or_goal": label},
                "reward": 0.0,
                "done": False,
                "info": {"episode_label": label},
            }
            for label in self.labels
        ]
        return rgb, states

    def force_stop_vector(self, reason):
        self.soft_stops += 1
        return self.reset_vector()

    def flush_logs_to_disk_slot(self, slot):
        return f"slot-{slot}"

    def soft_stop_count(self):
        return self.soft_stops


@ray.remote
class _WatchdogVLM:
    def __init__(self, episode_delay=0.0):
        self.episode_delay = float(episode_delay)
        self.labels = []

    def run_episode_batch(self, env, initial_state):
        time.sleep(self.episode_delay)
        self.labels = [state["info"]["episode_label"] for state in initial_state[1]]
        return False, [dict(state["info"], done=True) for state in initial_state[1]], {}

    def postprocess_batch(self, **kwargs):
        return [
            {
                "rewards": np.asarray([0.0], dtype=np.float32),
                "dones": np.asarray([True]),
                "episode_label": np.asarray([label]),
            }
            for label in self.labels
        ]

    def ping(self):
        return True


@ray.remote(max_restarts=0, max_task_retries=0)
class _ResetDeathEnv:
    def assign_shard(self, labels):
        self.labels = list(labels)

    def reset_vector(self):
        os._exit(17)


@ray.remote(max_restarts=0, max_task_retries=0)
class _EpisodeDeathEnv:
    def assign_shard(self, labels):
        self.labels = list(labels)

    def reset_vector(self):
        rgb = np.zeros((len(self.labels), 2, 2, 3), dtype=np.uint8)
        states = [
            {
                "obs": {"instr_or_goal": label},
                "reward": 0.0,
                "done": False,
                "info": {"episode_label": label},
            }
            for label in self.labels
        ]
        return rgb, states

    def die(self):
        os._exit(18)


@ray.remote(max_restarts=0, max_task_retries=0)
class _CrashOnceVLM:
    def __init__(self):
        self.calls = 0
        self.labels = []

    def run_episode_batch(self, env, initial_state):
        self.labels = [state["info"]["episode_label"] for state in initial_state[1]]
        if self.calls == 0:
            self.calls += 1
            ray.get(env.die.remote())
        return False, [dict(state["info"], done=True) for state in initial_state[1]], {}

    def postprocess_batch(self, **kwargs):
        return [
            {
                "rewards": np.asarray([0.0], dtype=np.float32),
                "dones": np.asarray([True]),
                "episode_label": np.asarray([label]),
            }
            for label in self.labels
        ]


@ray.remote
class _StepTimeoutOnceVLM:
    def __init__(self):
        self.calls = 0
        self.labels = []

    def run_episode_batch(self, env, initial_state):
        self.labels = [state["info"]["episode_label"] for state in initial_state[1]]
        if self.calls == 0:
            self.calls += 1
            raise RuntimeError(
                "NavVerse vector step exceeded 300s; "
                f"active_episodes={self.labels}"
            )
        return False, [dict(state["info"], done=True) for state in initial_state[1]], {}

    def postprocess_batch(self, **kwargs):
        return [
            {
                "rewards": np.asarray([0.0], dtype=np.float32),
                "dones": np.asarray([True]),
                "episode_label": np.asarray([label]),
            }
            for label in self.labels
        ]


@ray.remote
class _AlwaysStepTimeoutVLM:
    def __init__(self):
        self.labels = []

    def run_episode_batch(self, env, initial_state):
        self.labels = [state["info"]["episode_label"] for state in initial_state[1]]
        raise RuntimeError(
            "NavVerse vector step exceeded 300s; "
            f"active_episodes={self.labels}"
        )

    def discard_episode_batch(self):
        labels, self.labels = self.labels, []
        return [(None, None, None) for _ in labels]


@ray.remote
class _FatalTaskVLM:
    def run_episode_batch(self, env, initial_state):
        raise RuntimeError("unrelated model failure")


@ray.remote
class _SoftTerminalEnv:
    def step_vector(self, actions):
        time.sleep(0.08)
        rgb = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        return rgb, [
            {
                "obs": {"instr_or_goal": "goal"},
                "reward": 0.25,
                "done": False,
                "info": {"episode_label": "scene_a_0", "truncated": False},
            }
        ]

    def force_stop_vector(self, reason):
        rgb = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        return rgb, [
            {
                "obs": {"instr_or_goal": "goal"},
                "reward": 0.0,
                "done": True,
                "info": {
                    "episode_label": "scene_a_0",
                    "truncated": True,
                    "termination_reason": reason,
                    "external_timeout_stop": True,
                },
            }
        ]


class _BatchedVisionDispatchStub:
    _infer_slots_with_batched_vision = RLWorker._infer_slots_with_batched_vision
    _shadow_stop_decision = RLWorker._shadow_stop_decision
    _stop_target = RLWorker._stop_target

    def __init__(self):
        self.policy_head_config = {"type": "continuous"}
        self.rollout_config = {"temperature": 1.0}
        self.rl_batch_sequence_states = [
            {"slot": index, "turn": 0} for index in range(3)
        ]
        self.prepared_slots = []

    def _prepare_model_for_inference(self):
        pass

    def restore_sequence_state(self, state):
        self.current_state = dict(state)

    def capture_sequence_state(self):
        return dict(self.current_state)

    def _prepare_infer_inputs(self, messages, images, pos_id_kwargs):
        assert len(images) == 1
        assert pos_id_kwargs == {"mode": "standard"}
        slot = self.current_state["slot"]
        self.prepared_slots.append(slot)
        return {"slot": slot, "messages": messages}

    def batch_image_features(self, prepared_inputs):
        return [f"vision-{inputs['slot']}" for inputs in prepared_inputs]

    def _forward_infer_inputs(
        self, inputs, *, temperature, precomputed_image_features
    ):
        slot = inputs["slot"]
        assert self.current_state["slot"] == slot
        assert precomputed_image_features == f"vision-{slot}"
        assert temperature == 1.0
        self.current_state["turn"] += 1
        return {"h": np.asarray([slot], dtype=np.float32)}, None

    def _sample_action_for_state(self, policy_out, action_logprobs, state_dict, logs):
        slot = int(policy_out["h"][0])
        assert action_logprobs is None
        assert state_dict["slot"] == slot
        return (
            np.asarray([slot], dtype=np.float32),
            np.zeros((10, 3), dtype=np.float32),
            np.float32(0.0),
            np.asarray([0], dtype=np.int64),
            f"slot-{slot}",
        )


def test_batched_vision_dispatch_preserves_slot_state_and_order():
    worker = _BatchedVisionDispatchStub()
    rgb = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    states = [{"slot": index} for index in range(3)]
    messages = [[{"slot": index}] for index in range(3)]

    decisions = worker._infer_slots_with_batched_vision(
        [2, 0], rgb, states, messages
    )

    assert worker.prepared_slots == [2, 0]
    assert list(decisions) == [2, 0]
    assert decisions[2][4] == "slot-2"
    assert decisions[0][4] == "slot-0"
    assert worker.rl_batch_sequence_states == [
        {"slot": 0, "turn": 1},
        {"slot": 1, "turn": 0},
        {"slot": 2, "turn": 1},
    ]


class _LocalBatchWorker(EpisodeRolloutMixin):
    run_episode_batch = RLWorker.run_episode_batch
    _record_batch_transition = RLWorker._record_batch_transition

    def __init__(self):
        self.rollout_config = {
            "max_steps": 3,
            "episode_soft_timeout_seconds": 0.05,
            "temperature": 1.0,
            "convo_start_template": [],
            "convo_turn_template": [],
        }
        self.policy_head_config = {"type": "continuous"}

    @staticmethod
    def new_sequence_state():
        return {}

    @staticmethod
    def capture_sequence_state():
        return {}

    @staticmethod
    def _infer_batch_slot(index, rgb, state, messages):
        return (
            np.asarray([0.1, 0.0, 0.0], dtype=np.float32),
            np.asarray([[0.1, 0.0, 0.0]], dtype=np.float32),
            np.float32(-0.5),
            None,
            "0.100,0.000,0.000",
            {},
            0.0,
        )


@pytest.mark.parametrize(
    ("reason", "native_done", "reached", "expected"),
    [
        ("bad_orientation", True, False, (False, True)),
        ("terrain_out_of_bounds", True, False, (False, True)),
        ("low_level_done", True, False, (False, True)),
        (None, True, False, (False, True)),
        ("time_out", True, False, (True, False)),
        ("stop_called", True, False, (False, False)),
        ("bad_orientation", True, True, (False, False)),
        (None, False, False, (False, False)),
    ],
)
def test_native_termination_classification(reason, native_done, reached, expected):
    assert _native_termination_flags(
        reason, native_done=native_done, reached=reached
    ) == expected


def test_host_episode_pool_excludes_only_configured_labels(tmp_path):
    pool_path = tmp_path / "episodes.txt"
    pool_path.write_text("scene_a_0\nscene_a_1\nscene_b_0\n")
    host = object.__new__(NavVerseHostActor)
    host._pool = None
    host._episodes_path = str(pool_path)
    host._train_uids = None
    host._excluded_episode_labels = {"scene_a_1"}

    host._load_pool()

    assert host._pool == ["scene_a_0", "scene_b_0"]


def test_host_episode_pool_rejects_unknown_exclusion(tmp_path):
    pool_path = tmp_path / "episodes.txt"
    pool_path.write_text("scene_a_0\n")
    host = object.__new__(NavVerseHostActor)
    host._pool = None
    host._episodes_path = str(pool_path)
    host._train_uids = None
    host._excluded_episode_labels = {"scene_missing_0"}

    with pytest.raises(ValueError, match="absent from the pool"):
        host._load_pool()


def test_host_training_rounds_use_train_uids_but_pool_stays_complete(tmp_path):
    pool = [f"scene_{scene}_{episode}" for scene in "abc" for episode in range(8)]
    train = [label for label in pool if label.startswith("scene_a_")]
    pool_path = tmp_path / "episodes.txt"
    train_path = tmp_path / "train.txt"
    pool_path.write_text("\n".join(pool) + "\n")
    train_path.write_text("\n".join(train) + "\n")

    host = object.__new__(NavVerseHostActor)
    host.n = 8
    host._pool = None
    host._episodes_path = str(pool_path)
    host._train_uids = frozenset(train)
    host._excluded_episode_labels = set()
    host._rng = np.random.default_rng(0)

    host._load_pool()
    host._build_scene_rounds()

    assert host._pool == pool
    assert len(host._scene_rounds) == 1
    assert set(host._scene_rounds[0]) == set(train)


def test_host_media_toggle_discards_training_frames():
    host = object.__new__(NavVerseHostActor)
    host.n = 2
    host.minimal_logging = False
    host.policy_camera_only = False
    host._frames = [[np.zeros((2, 2, 3), dtype=np.uint8)], [np.zeros((2, 2, 3), dtype=np.uint8)]]

    host.set_media_enabled(False)

    assert host.minimal_logging is True
    assert host._frames == [[], []]

    host.set_media_enabled(True)
    assert host.minimal_logging is False


def test_host_media_toggle_enables_third_person_only_for_eval():
    camera_names = []
    host = object.__new__(NavVerseHostActor)
    host.n = 2
    host.minimal_logging = True
    host.policy_camera_only = True
    host._frames = [[], []]
    host.vln_sim = types.SimpleNamespace(
        set_on_demand_camera_names=lambda names: camera_names.append(tuple(names))
    )

    host.set_media_enabled(True)
    host.set_media_enabled(False)

    assert camera_names == [
        ("pov_camera", "third_person_camera"),
        ("pov_camera",),
    ]


def test_host_force_stop_is_terminal_and_idempotent():
    host = object.__new__(NavVerseHostActor)
    host.n = 2
    host._isaac_thread_id = threading.get_ident()
    host.success_distance = 1.6
    host._done = [False, False]
    host._episode_label = ["scene_a_0", "scene_a_1"]
    host._prev_distance = [2.0, 3.0]
    host._start_distance = [4.0, 5.0]
    host._min_distance = [2.0, 3.0]
    host._path_length = [1.0, 2.0]
    host._steps = [7, 8]
    host._infos = [[{"episode_label": "scene_a_0"}], [{"episode_label": "scene_a_1"}]]
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    host._vector_results = {
        index: (
            rgb.copy(),
            {
                "obs": {"instr_or_goal": "goal"},
                "reward": 0.1,
                "done": False,
                "info": {"episode_label": host._episode_label[index]},
            },
        )
        for index in range(2)
    }

    _, states = host.force_stop_vector("wall_timeout_soft_stop")
    assert host._done == [True, True]
    assert all(state["done"] for state in states)
    assert all(state["info"]["truncated"] for state in states)
    assert all(
        state["info"]["termination_reason"] == "wall_timeout_soft_stop"
        for state in states
    )
    info_lengths = [len(items) for items in host._infos]
    host.force_stop_vector("wall_timeout_soft_stop")
    assert [len(items) for items in host._infos] == info_lengths


def test_worker_soft_stop_marks_last_real_transition_without_adding_action(ray_session):
    worker = _LocalBatchWorker()
    env = _SoftTerminalEnv.remote()
    initial = (
        np.zeros((1, 2, 2, 3), dtype=np.uint8),
        [
            {
                "obs": {"instr_or_goal": "goal"},
                "reward": 0.0,
                "done": False,
                "info": {"episode_label": "scene_a_0"},
            }
        ],
    )

    _, results, _ = worker.run_episode_batch(env, initial)
    trajectory = worker.rl_batch_trajectories[0]
    assert trajectory["actions_continuous"].shape[0] == 1
    assert trajectory["dones"].tolist() == [True]
    assert trajectory["truncated"].tolist() == [True]
    assert trajectory["termination_reason"].tolist() == ["wall_timeout_soft_stop"]
    assert results[0]["termination_reason"] == "wall_timeout_soft_stop"


def test_soft_deadline_requests_stop_without_rebuild(ray_session):
    env = _WatchdogEnv.remote()
    vlm = _WatchdogVLM.remote(episode_delay=0.15)
    envs = [env]
    rebuilds = []
    ray.get([env.soft_stop_count.remote(), vlm.ping.remote()])

    def rebuild(index):
        rebuilds.append(index)
        return _WatchdogEnv.remote()

    rollouts, results, _ = collect_vector_rollouts(
        envs,
        [vlm],
        iter([["scene_a_0", "scene_a_1"]]),
        target_episodes=2,
        episodes_per_worker=2,
        episode_soft_timeout_seconds=0.05,
        episode_hard_timeout_seconds=0.5,
        sim_restart_limit=1,
        sim_rebuilder=rebuild,
    )

    assert len(rollouts) == len(results) == 2
    assert ray.get(env.soft_stop_count.remote()) == 1
    assert rebuilds == []


def test_hard_deadline_rebuilds_and_retries_exact_shard(ray_session):
    old_env = _WatchdogEnv.remote(reset_delay=5.0)
    replacement = _WatchdogEnv.options(num_cpus=0).remote()
    vlm = _WatchdogVLM.remote()
    envs = [old_env]
    rebuilds = []
    validated = []
    ray.get(
        [
            old_env.soft_stop_count.remote(),
            replacement.soft_stop_count.remote(),
            vlm.ping.remote(),
        ]
    )

    def rebuild(index):
        rebuilds.append(index)
        return replacement

    rollouts, results, _ = collect_vector_rollouts(
        envs,
        [vlm],
        iter([["scene_a_0", "scene_a_1"]]),
        target_episodes=2,
        episodes_per_worker=2,
        episode_soft_timeout_seconds=0.2,
        episode_hard_timeout_seconds=1.0,
        sim_restart_limit=1,
        sim_rebuilder=rebuild,
        sim_rebuild_validator=lambda index, handle: validated.append((index, handle)),
    )

    assert len(rollouts) == 2
    assert [result["episode_label"] for result in results] == [
        "scene_a_0",
        "scene_a_1",
    ]
    assert rebuilds == [0]
    assert validated == [(0, replacement)]
    assert envs[0] != old_env


def test_replacement_failure_stops_after_configured_limit(ray_session):
    envs = [_WatchdogEnv.remote(reset_delay=5.0)]
    replacement = _WatchdogEnv.options(num_cpus=0).remote(reset_delay=5.0)
    vlm = _WatchdogVLM.remote()
    rebuilds = []
    ray.get(
        [
            envs[0].soft_stop_count.remote(),
            replacement.soft_stop_count.remote(),
            vlm.ping.remote(),
        ]
    )

    def rebuild(index):
        rebuilds.append(index)
        return replacement

    with pytest.raises(RuntimeError, match="sim_restart_limit=1"):
        collect_vector_rollouts(
            envs,
            [vlm],
            iter([["scene_a_0", "scene_a_1"]]),
            target_episodes=2,
            episodes_per_worker=2,
            episode_soft_timeout_seconds=0.2,
            episode_hard_timeout_seconds=1.0,
            sim_restart_limit=1,
            sim_rebuilder=rebuild,
        )
    assert rebuilds == [0]


def test_actor_death_during_reset_rebuilds_and_retries_shard(ray_session):
    replacement = _WatchdogEnv.options(num_cpus=0).remote()
    envs = [_ResetDeathEnv.remote()]
    vlm = _WatchdogVLM.remote()
    rebuilds = []

    def rebuild(index):
        rebuilds.append(index)
        return replacement

    rollouts, results, _ = collect_vector_rollouts(
        envs,
        [vlm],
        iter([["scene_a_0", "scene_a_1"]]),
        target_episodes=2,
        episodes_per_worker=2,
        sim_restart_limit=1,
        sim_rebuilder=rebuild,
    )

    assert len(rollouts) == len(results) == 2
    assert rebuilds == [0]
    assert envs == [replacement]


def test_actor_death_inside_vlm_task_rebuilds_sim_and_retries_shard(ray_session):
    replacement = _WatchdogEnv.options(num_cpus=0).remote()
    envs = [_EpisodeDeathEnv.remote()]
    vlm = _CrashOnceVLM.remote()
    rebuilds = []

    def rebuild(index):
        rebuilds.append(index)
        return replacement

    rollouts, results, _ = collect_vector_rollouts(
        envs,
        [vlm],
        iter([["scene_a_0", "scene_a_1"]]),
        target_episodes=2,
        episodes_per_worker=2,
        sim_restart_limit=1,
        sim_rebuilder=rebuild,
    )

    assert len(rollouts) == len(results) == 2
    assert rebuilds == [0]
    assert envs == [replacement]


def test_vector_step_timeout_rebuilds_sim_and_retries_exact_shard(ray_session):
    envs = [_WatchdogEnv.remote()]
    replacement = _WatchdogEnv.options(num_cpus=0).remote()
    vlm = _StepTimeoutOnceVLM.remote()
    rebuilds = []

    def rebuild(index):
        rebuilds.append(index)
        return replacement

    rollouts, results, _ = collect_vector_rollouts(
        envs,
        [vlm],
        iter([["scene_a_0", "scene_a_1"]]),
        target_episodes=2,
        episodes_per_worker=2,
        sim_restart_limit=1,
        sim_rebuilder=rebuild,
    )

    assert len(rollouts) == len(results) == 2
    assert [result["episode_label"] for result in results] == [
        "scene_a_0",
        "scene_a_1",
    ]
    assert rebuilds == [0]
    assert envs == [replacement]


def test_repeated_vector_step_timeout_discards_shard_and_keeps_rebuilt_sim(ray_session):
    envs = [_WatchdogEnv.remote()]
    replacements = [
        _WatchdogEnv.options(num_cpus=0).remote(),
        _WatchdogEnv.options(num_cpus=0).remote(),
    ]
    vlm = _AlwaysStepTimeoutVLM.remote()
    rebuilds = []
    validated = []
    timing = {}

    def rebuild(index):
        rebuilds.append(index)
        return replacements[len(rebuilds) - 1]

    rollouts, results, logs = collect_vector_rollouts(
        envs,
        [vlm],
        iter([["scene_a_0", "scene_a_1"]]),
        target_episodes=2,
        episodes_per_worker=2,
        sim_restart_limit=1,
        sim_rebuilder=rebuild,
        sim_rebuild_validator=lambda index, handle: validated.append((index, handle)),
        timing_out=timing,
    )

    assert rollouts == [(None, None, None), (None, None, None)]
    assert [result["termination_reason"] for result in results] == [
        "sim_vector_step_timeout",
        "sim_vector_step_timeout",
    ]
    assert ray.get(logs) == [None, None]
    assert rebuilds == [0, 0]
    assert validated == [(0, replacements[0]), (0, replacements[1])]
    assert envs == [replacements[1]]
    assert timing["runtime/sim_timeout_abandoned_shards"] == 1.0
    assert timing["runtime/sim_timeout_abandoned_episodes"] == 2.0


def test_unrelated_vlm_task_error_is_not_retried(ray_session):
    envs = [_WatchdogEnv.remote()]
    rebuilds = []

    with pytest.raises(ray.exceptions.RayTaskError, match="unrelated model failure"):
        collect_vector_rollouts(
            envs,
            [_FatalTaskVLM.remote()],
            iter([["scene_a_0", "scene_a_1"]]),
            target_episodes=2,
            episodes_per_worker=2,
            sim_restart_limit=1,
            sim_rebuilder=lambda index: rebuilds.append(index),
        )

    assert rebuilds == []


def test_cycle_boundary_recycling_is_sequential(monkeypatch):
    old_sims = [object(), object()]
    original_sims = list(old_sims)
    new_sims = [object(), object()]
    events = []
    monkeypatch.setattr(
        "longnav.utils.train_loop.ray.kill",
        lambda actor, no_restart: events.append(("kill", actor, no_restart)),
    )

    def rebuild(index):
        events.append(("rebuild", index))
        return new_sims[index]

    def validate(index, actor):
        events.append(("validate", index, actor))

    recycle_vector_sims(old_sims, rebuild, validate)

    assert old_sims == new_sims
    assert events == [
        ("kill", original_sims[0], True),
        ("rebuild", 0),
        ("validate", 0, new_sims[0]),
        ("kill", original_sims[1], True),
        ("rebuild", 1),
        ("validate", 1, new_sims[1]),
    ]
