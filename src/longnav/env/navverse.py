"""NavVerse (Isaac Lab) continuous-action env actor for flowsde-branch RL.

Executes flow-SDE / continuous-policy action chunks -- (gap, 3) cumulative
body-frame SE(2) setpoints, the SAME convention the continuous_flow benchmark
backend uses (baselines/longnav/longnav_agent.py:relative_se2_to_world) -- as
one TIMED_TRAJECTORY per policy step, driven through NavVerse's own velocity
waypoint follower (navverse.sim.VLNSim._handle_timed_trajectory_command /
follow_timed_trajectory: linear SE(2) interpolation + PID-style tracking).
That is the exact mechanism the continuous eval harness drives over the wire,
reused in-process here so training and eval execute chunks identically.

Ported from the discrete `NavverseEnv` in `baselines/longnav/spatial_training_rl`.
Isaac Lab bootstrap and VLNSim wiring are retained, while the action is changed
from a discrete command to a continuous SE(2) chunk. The production topology
also retains the previous NavVerse batching scheme: each simulator actor owns
eight vectorized robots and is paired with one VLM actor on the same GPU. The
single-environment actor and slot-proxy adapter remain available for focused
smoke tests, but the 8-GPU training path dispatches complete host-sized waves.

No stop head: like `objectnav_continuous.ContinuousObjectNavEnvActor`, reward
is clipped geodesic progress per step with no terminal success bonus by
default -- the continuous checkpoints this branch trains have no STOP head, so
a success bonus would optimize a termination heuristic the policy cannot even
choose. Termination is env-driven: goal reached (distance_to_goal <=
success_distance), step budget spent, or the simulator's own termination
manager firing (episode timeout, robot fell out of the world, etc).

Episode pool ordering is SCENE-MAJOR, not a flat shuffle: `assign_shard(None)`
(the framework's trivial-shard training default) shuffles the scene order and,
within each scene, the episode order, but never interleaves two scenes' rows.
A NavVerse/Isaac Lab scene switch reloads the whole USD stage -- expensive
enough that `reset_batch` in the old vectorized env refused to mix scenes
within one batch at all. A flat shuffle across scenes would reload the scene
almost every single episode; grouping keeps that cost to once per
`episodes_path`'s per-scene run length, independent of how many parallel
actors happen to be running.
"""

from __future__ import annotations

import asyncio
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import ray
import torch


def _yaw_from_xyzw(quat_xyzw) -> float:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _relative_se2_to_world(position, quat_xyzw, chunk: np.ndarray) -> np.ndarray:
    """Cumulative body-frame [dx, dy, dtheta] rows -> world SE(2) waypoints.

    Verbatim convention of `longnav_agent.relative_se2_to_world`: every row is
    relative to the SAME pose (the pose at the start of the chunk), not to the
    previous row. Re-deriving this independently would silently disagree with
    the benchmark harness the checkpoint was trained/evaluated against.
    """
    base = np.asarray(position, dtype=np.float64)
    yaw = _yaw_from_xyzw(quat_xyzw)
    c, s = math.cos(yaw), math.sin(yaw)
    world = np.empty_like(chunk, dtype=np.float64)
    for i, (dx, dy, dtheta) in enumerate(chunk):
        world_yaw = (yaw + dtheta + math.pi) % (2.0 * math.pi) - math.pi
        world[i] = (
            base[0] + c * dx - s * dy,
            base[1] + s * dx + c * dy,
            world_yaw,
        )
    return world


def _scene_of(label: str) -> str:
    """`sceneName_episodeIndex` -> `sceneName` (the episodes/*.txt label convention)."""
    return label.rsplit("_", 1)[0]


def _scene_request_is_ready(vln_sim, label: str) -> bool:
    current_episode = getattr(vln_sim, "current_episode", None)
    current_label = getattr(current_episode, "episode_label", None)
    return bool(
        not getattr(vln_sim, "reset_flag", False)
        and getattr(vln_sim, "sim_state", None) == "running"
        and current_label == label
    )


def _uid_filter(value: Optional[Any]) -> Optional[frozenset[str]]:
    if not value:
        return None
    if isinstance(value, str):
        with open(value) as stream:
            value = stream.read().replace("\n", ",").split(",")
    return frozenset(str(uid).strip() for uid in value if str(uid).strip())


def _native_termination_flags(
    termination_reason: Optional[str], *, native_done: bool, reached: bool
) -> tuple[bool, bool]:
    """Return `(truncated, failed)` for a simulator-originated episode ending."""
    native_truncated = bool(native_done and not reached and termination_reason == "time_out")
    native_failure = bool(
        native_done
        and not reached
        and termination_reason not in {"time_out", "stop_called"}
    )
    return native_truncated, native_failure


def _set_robot_visuals_hidden(vln_sim, hidden: bool) -> int:
    """Toggle robot render geometry without changing physics or sensor prims."""
    from isaaclab.sim import utils as prim_utils

    manager_env = getattr(vln_sim, "manager_env", None)
    if manager_env is None:
        raise RuntimeError("Robot visuals cannot be changed before the scene is initialized")
    stage = manager_env.scene.stage
    changed = 0
    for env_index in range(int(manager_env.num_envs)):
        robot = stage.GetPrimAtPath(f"/World/envs/env_{env_index}/Robot")
        if not robot.IsValid():
            continue
        prim_utils.set_prim_visibility(robot, not hidden)
        changed += 1
    if hidden and changed == 0:
        raise RuntimeError("hide_robot_visuals found no robot render geometry")
    return changed


def _install_manager_profile_hooks(vln_sim, stats: Dict[str, tuple]) -> None:
    manager_env = vln_sim.manager_env
    hooks = (
        (manager_env.action_manager, "process_action", "action/process"),
        (manager_env.action_manager, "apply_action", "action/apply"),
        (manager_env.scene, "write_data_to_sim", "scene/write"),
        (manager_env.sim, "step", "sim/physics_step"),
        (manager_env.sim, "render", "sim/render"),
        (manager_env.scene, "update", "scene/update"),
        (manager_env.observation_manager, "compute", "manager/observation"),
        (manager_env.termination_manager, "compute", "manager/termination"),
        (manager_env.reward_manager, "compute", "manager/reward"),
        (vln_sim.env.termination_manager, "compute", "wrapper/termination"),
    )
    for owner, method_name, profile_key in hooks:
        original = getattr(owner, method_name)

        def profiled(*args, _original=original, _key=profile_key, **kwargs):
            started = time.perf_counter()
            try:
                return _original(*args, **kwargs)
            finally:
                total_s, count = stats.get(_key, (0.0, 0))
                stats[_key] = (total_s + time.perf_counter() - started, count + 1)

        setattr(owner, method_name, profiled)


def _collect_profile_stats(
    vln_sim, manager_stats, *, reset: bool, host_stats=None
) -> Dict[str, Any]:
    profile = {}
    sources = (
        ("manager", manager_stats),
        ("sim", vln_sim.profile_stats),
        ("wrapper", vln_sim.env.profile_stats),
        ("host", host_stats or {}),
    )
    for prefix, stats in sources:
        for key, (total_s, count) in stats.items():
            profile[f"{prefix}/{key}"] = {
                "total_seconds": float(total_s),
                "count": int(count),
                "mean_seconds": float(total_s) / max(int(count), 1),
            }
        if reset:
            stats.clear()
    return profile


class NavVerseEnvActor:
    """Five-method env-actor interface, over the NavVerse/Isaac-Lab simulator.

    `reset()` / `step(action)` return `(rgb, state_dict)` with the `obs` /
    `reward` / `done` / `is_exhausted` / `info` keys `rollout_core` consumes.
    `action` is a `(gap, 3)` chunk of cumulative body-frame SE(2) setpoints.
    """

    def __init__(
        self,
        navverse_repo_root: str = "/home/ubuntu/Projects/NavVerse-Benchmark",
        config_path: str = "configs/default.yaml",
        episode_folder: str = "/home/ubuntu/Projects/navverse_data/episodes/",
        scene_folder: str = "/home/ubuntu/Projects/navverse_data/",
        episodes_path: Optional[str] = None,
        train_uids: Optional[Any] = None,
        excluded_episode_labels: Optional[List[str]] = None,
        task_type: str = "placenav",
        robot_name: str = "spot",
        tidybot_embodiment: str = "full",
        hide_robot_visuals: bool = False,
        on_demand_render: bool = False,
        camera_rgb_only: bool = False,
        policy_camera_only: bool = False,
        disable_camera: bool = False,
        trajectory_planner: Optional[str] = None,
        profile_step_timing: bool = False,
        profile_step_interval: int = 100,
        gap: int = 10,
        dt: float = 0.04,
        max_steps: int = 175,
        success_distance: float = 1.6,
        slack_penalty: float = 0.0,
        collision_penalty: float = 0.0,
        failure_penalty: float = 0.0,
        progress_reward_clip: float = 0.75,
        success_reward: float = 0.0,
        timeout_margin_s: float = 1.0,
        minimal_logging: bool = False,
        video_fps: int = 2,
        video_tick_stride: int = 0,
        video_realtime_factor: float = 1.0,
        logging_output_dir: Optional[str] = None,
        logger_actor: Any = None,
        **kwargs: Any,
    ):
        self.gap = int(gap)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.success_distance = float(success_distance)
        self.slack_penalty = float(slack_penalty)
        self.collision_penalty = float(collision_penalty)
        self.failure_penalty = float(failure_penalty)
        self.progress_reward_clip = float(progress_reward_clip)
        self.success_reward = float(success_reward)
        self.timeout_margin_s = float(timeout_margin_s)
        self.minimal_logging = bool(minimal_logging)
        self.policy_camera_only = bool(policy_camera_only)
        self.video_fps = int(video_fps)
        self.video_tick_stride = int(video_tick_stride)
        self.video_realtime_factor = float(video_realtime_factor)
        if self.video_tick_stride < 0:
            raise ValueError("video_tick_stride must be non-negative")
        if self.video_realtime_factor <= 0:
            raise ValueError("video_realtime_factor must be positive")
        self.logging_output_dir = logging_output_dir
        self.logger_actor = logger_actor
        self._log_prefix = ""

        self._episodes_path = episodes_path
        self._train_uids = _uid_filter(train_uids)
        self._excluded_episode_labels = set(excluded_episode_labels or [])
        self._pool: Optional[List[str]] = None
        self._shard: Optional[List[str]] = None
        self._order: List[str] = []
        self._cursor = 0
        self._rng = np.random.default_rng(os.getpid())
        self._steps = 0
        self._prev_distance: Optional[float] = None
        self._start_distance: Optional[float] = None
        self._min_distance: Optional[float] = None
        self._path_length = 0.0
        self._prev_pose: Optional[tuple] = None
        self._episode = None
        self._frames: List[np.ndarray] = []
        self._infos: List[Dict[str, Any]] = []
        self._rewards: List[float] = []

        self.manager_profile_stats: Dict[str, tuple] = {}
        self.manager_profile_hooks_installed = False
        self.profile_step_timing = bool(profile_step_timing)
        self.profile_step_interval = int(profile_step_interval)

        self._boot_isaac_lab(
            navverse_repo_root=navverse_repo_root,
            config_path=config_path,
            episode_folder=episode_folder,
            scene_folder=scene_folder,
            task_type=task_type,
            robot_name=robot_name,
            tidybot_embodiment=tidybot_embodiment,
            hide_robot_visuals=hide_robot_visuals,
            on_demand_render=on_demand_render,
            camera_rgb_only=camera_rgb_only,
            policy_camera_only=policy_camera_only,
            disable_camera=disable_camera,
            trajectory_planner=trajectory_planner,
        )

    # -- construction, done lazily inside the actor process ------------------------------
    # Ray pickles constructor kwargs to the worker process and instantiates there, so the
    # Isaac Lab app / VLNSim (neither picklable) are only ever built inside __init__, which
    # already runs inside that worker -- no extra lazy-init indirection needed here, unlike
    # the habitat continuous actor (whose constructor args are otherwise picklable but whose
    # Simulator object is not built until first use).
    def _boot_isaac_lab(
        self,
        *,
        navverse_repo_root,
        config_path,
        episode_folder,
        scene_folder,
        task_type,
        robot_name,
        tidybot_embodiment,
        hide_robot_visuals,
        on_demand_render,
        camera_rgb_only,
        policy_camera_only,
        disable_camera,
        trajectory_planner,
    ):
        def is_linux_headless():
            has_x11 = "DISPLAY" in os.environ
            has_wayland = "WAYLAND_DISPLAY" in os.environ
            return not (has_x11 or has_wayland)

        # NavVerse resolves its Hydra config/episode/scene paths relative to CWD (see
        # navverse/config.py:load_config), which is the main NavVerse-Benchmark repo root
        # in every other invocation of this codebase -- but a Ray actor inherits whatever
        # CWD the driver script happened to be run from (this RL trainer's own directory),
        # so without this chdir NavVerse's own `configs/default.yaml` lookup 404s.
        os.chdir(navverse_repo_root)

        import argparse
        import sys

        from isaaclab.app import AppLauncher
        import navverse.utils.rsl_rl_cli_args as rsl_rl_cli_args
        import navverse.vln_args as vln_cli_args

        parser = argparse.ArgumentParser(description="NavVerse RL env actor")
        rsl_rl_cli_args.add_rsl_rl_args(parser)
        vln_cli_args.add_vln_args(parser)
        AppLauncher.add_app_launcher_args(parser)

        original_argv = sys.argv.copy()
        navverse_args = [
            "--episode_folder", str(episode_folder),
            "--scene_folder", str(scene_folder),
            "--task_type", str(task_type),
            "--robot_name", str(robot_name),
            "--tidybot_embodiment", str(tidybot_embodiment),
            "--timed_trajectory_control_dt", str(self.dt),
            "--on_demand_render", str(bool(on_demand_render)),
            "--disable_camera", str(bool(disable_camera)),
            "--profile_step_timing", str(self.profile_step_timing),
            "--profile_step_interval", str(self.profile_step_interval),
            "--num_envs", "1",
        ]
        if str(config_path).lower() != "auto":
            navverse_args.extend(["--config", str(config_path)])
        if trajectory_planner:
            navverse_args.extend(["--trajectory_planner", str(trajectory_planner)])

        sys.argv = [sys.argv[0], *navverse_args]
        args = vln_cli_args.parse_args(parser)
        sys.argv = original_argv
        args.disable_socket_server = True
        args.headless = is_linux_headless()

        if camera_rgb_only:
            os.environ["NAVVERSE_CAMERA_RGB_ONLY"] = "1"
        os.environ["NAVVERSE_ON_DEMAND_CAMERAS"] = (
            "pov_camera" if policy_camera_only else "pov_camera,third_person_camera"
        )
        app_launcher = AppLauncher(args)
        self.simulation_app = app_launcher.app

        import omni.kit.app

        omni.kit.app.get_app().update()
        from navverse.sim import VLNSim

        self.vln_sim = VLNSim(args)
        self._hide_robot_visuals = bool(hide_robot_visuals)
        self.action_names = ["STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT"]

    def _install_manager_profile_hooks(self):
        if self.manager_profile_hooks_installed:
            return
        _install_manager_profile_hooks(self.vln_sim, self.manager_profile_stats)
        self.manager_profile_hooks_installed = True

    def get_vector_profile(self, reset=True):
        return _collect_profile_stats(
            self.vln_sim, self.manager_profile_stats, reset=bool(reset)
        )

    # -- env-actor interface ---------------------------------------------------------------
    def set_log_prefix(self, prefix: str) -> None:
        self._log_prefix = str(prefix or "")

    def set_media_enabled(self, enabled: bool) -> None:
        self.minimal_logging = not bool(enabled)
        if self.policy_camera_only:
            self.vln_sim.set_on_demand_camera_names(
                ("pov_camera", "third_person_camera")
                if enabled
                else ("pov_camera",)
            )
        if self.minimal_logging:
            self._frames = []

    def _load_pool(self) -> None:
        """Lazily read this actor's configured episode-label file (one label per line, the
        same `episodes/*.txt` convention `navverse_tools`/the benchmark harness already use
        -- train_set.txt, test_set.txt, warmup_set.txt, a sampled subset, etc). Loading here
        rather than in `__init__` matches the pattern the other real envs use for anything
        that touches disk/the sim: constructor kwargs get pickled to the Ray worker, so
        anything lazy is cheaper to retry and easier to reason about than doing it before the
        actor exists.
        """
        if self._pool is not None:
            return
        if not self._episodes_path:
            raise ValueError(
                "NavVerseEnvActor has no episodes_path configured and assign_shard(None) "
                "('trivial shard', the training default) needs one to know its full pool."
            )
        with open(self._episodes_path) as f:
            self._pool = [line.strip() for line in f if line.strip()]
        missing = self._excluded_episode_labels.difference(self._pool)
        if missing:
            raise ValueError(f"Excluded episode labels are absent from the pool: {sorted(missing)}")
        if self._excluded_episode_labels:
            self._pool = [label for label in self._pool if label not in self._excluded_episode_labels]
            print(f"[navverse] excluded {len(self._excluded_episode_labels)} invalid episode(s)")
        if not self._pool:
            raise ValueError(f"episodes_path={self._episodes_path} resolved to zero labels")
        if self._train_uids is not None:
            missing_train = self._train_uids.difference(self._pool)
            if missing_train:
                raise ValueError(
                    f"train_uids contains labels absent from the pool: {sorted(missing_train)}"
                )
            print(
                f"[navverse] train_uids: serving {len(self._train_uids)} of "
                f"{len(self._pool)} episodes for training"
            )

    def _reshuffle(self) -> None:
        """Shuffle SCENES, then shuffle each scene's episodes internally, then concatenate --
        never interleave two scenes' rows. A NavVerse/Isaac-Lab scene switch reloads the
        whole USD stage (the old vectorized env's `reset_batch` refused to even mix scenes
        within one batch, for the same reason); a flat shuffle across scenes would pay that
        reload on nearly every episode, so grouping keeps the reload cost to once per scene's
        run length regardless of how many parallel actors happen to be running.
        """
        labels = self._shard if self._shard is not None else (
            [label for label in self._pool if label in self._train_uids]
            if self._train_uids is not None
            else self._pool
        )
        by_scene: Dict[str, List[str]] = {}
        for label in labels:
            by_scene.setdefault(_scene_of(label), []).append(label)
        scenes = list(by_scene.keys())
        self._rng.shuffle(scenes)
        order: List[str] = []
        for scene in scenes:
            eps = list(by_scene[scene])
            self._rng.shuffle(eps)
            order.extend(eps)
        self._order = order
        self._cursor = 0

    def assign_shard(self, episodes: Optional[List[str]] = None) -> None:
        """`None` (the training default, via the framework's trivial-shard convention) means
        serve this actor's full configured pool (`episodes_path`); an explicit list restricts
        it to those labels (eval use case) -- never "serve nothing", which would read as an
        already-exhausted actor and silently drop every episode assigned that way.
        """
        self._load_pool()
        new_shard = None if episodes is None else list(episodes)
        if self._shard is not None and new_shard == self._shard:
            self._reshuffle()  # same shard handed back: fresh permutation, not a reparse
            return
        self._shard = new_shard
        self._reshuffle()

    def is_exhausted(self) -> bool:
        return self._cursor >= len(self._order)

    def _next_label(self) -> str:
        label = self._order[self._cursor]
        self._cursor += 1
        return label

    # -- sim plumbing shared with the discrete env's single-episode path -------------------
    def _sim_time(self) -> float:
        manager_env = getattr(self.vln_sim, "manager_env", None)
        if manager_env is None:
            return 0.0
        return float(getattr(manager_env, "common_step_counter", 0) or 0) * float(
            getattr(manager_env, "step_dt", 0.0) or 0.0
        )

    def _latest_state_tuple(self, default_reward=0.0, default_done=False):
        obs = self.vln_sim.obs
        reward = self.vln_sim.reward
        done = self.vln_sim.done
        info = self.vln_sim.info or {}
        if reward is None:
            reward = default_reward
        if done is None:
            done = default_done
        return obs, reward, done, info

    def _step_until(self, predicate, timeout_sim_s=None, on_timeout=None, capture_video=False):
        start_sim_time = self._sim_time()
        tick = 0
        while True:
            self.vln_sim.step()
            tick += 1
            if (
                capture_video
                and self.video_tick_stride > 0
                and tick % self.video_tick_stride == 0
            ):
                obs = self.vln_sim.obs
                if obs is not None:
                    self._frames.append(self._rgb(obs).copy())
            if predicate():
                break
            if self.vln_sim.sim_state == "terminated":
                break
            if timeout_sim_s is not None and self._sim_time() - start_sim_time >= timeout_sim_s:
                if on_timeout is not None:
                    on_timeout()
                break
        return self._latest_state_tuple()

    def _robot_pose(self):
        robot = self.vln_sim.env.scene["robot"]
        root = robot.data.root_state_w[0, :7].detach().cpu().numpy()
        position = root[:3].astype(np.float64)
        quat_wxyz = root[3:7].astype(np.float64)
        quat_xyzw = np.concatenate([quat_wxyz[1:], quat_wxyz[:1]])
        return position, quat_xyzw

    @staticmethod
    def _find_obs_value(obs, key):
        if not hasattr(obs, "keys"):
            return None
        keys = list(obs.keys())
        if key in keys:
            return obs[key]
        for obs_key in keys:
            found = NavVerseEnvActor._find_obs_value(obs[obs_key], key)
            if found is not None:
                return found
        return None

    def _rgb(self, obs) -> np.ndarray:
        pov_rgb = self._find_obs_value(obs, "pov_rgb")
        if pov_rgb is None:
            raise KeyError("pov_rgb not found in NavVerse obs")
        return pov_rgb.squeeze().detach().cpu().numpy()[..., :3].astype(np.uint8)

    def _instruction(self) -> str:
        episode = self.vln_sim.current_episode or {}
        return (
            episode.get("placenav_goal")
            or episode.get("instruction")
            or episode.get("objnav")
            or ""
        )

    @staticmethod
    def _distance_to_goal(episode, position):
        """Reference-path distance used by the existing vectorized NavVerse RL env."""
        best_distance = float("inf")
        closest_goal_idx = 0
        for goal_idx, goal in enumerate(episode["goals"]):
            reference_path = goal.get("reference_path") or [goal["location"]]
            remaining = 0.0
            for idx in range(len(reference_path) - 1, -1, -1):
                if idx < len(reference_path) - 1:
                    remaining += float(
                        np.linalg.norm(
                            np.asarray(reference_path[idx], dtype=np.float64)
                            - np.asarray(reference_path[idx + 1], dtype=np.float64)
                        )
                    )
                candidate = float(
                    np.linalg.norm(
                        np.asarray(position, dtype=np.float64)
                        - np.asarray(reference_path[idx], dtype=np.float64)
                    )
                ) + remaining
                if candidate < best_distance:
                    best_distance = candidate
                    closest_goal_idx = goal_idx
        radius = float(episode["goals"][closest_goal_idx].get("radius", 0.0))
        return max(1e-5, best_distance - radius), closest_goal_idx

    # -- five-method interface ---------------------------------------------------------------
    def reset(self):
        if self.is_exhausted():
            raise RuntimeError("NavVerseEnvActor reset called after episode shard was exhausted")
        episode_label = self._next_label()
        self.vln_sim.load_episode(episode_label)
        obs, _, _, info = self._step_until(lambda: self.vln_sim.sim_state == "running")
        if self._hide_robot_visuals:
            hidden_count = _set_robot_visuals_hidden(self.vln_sim, True)
            print(f"[navverse] hidden robot visual prims: {hidden_count}")
        if self.profile_step_timing:
            self._install_manager_profile_hooks()

        measurements = (info or {}).get("measurements", info or {})
        distance = measurements.get("distance_to_goal")
        self._prev_distance = float(distance) if distance is not None else None
        self._start_distance = self._prev_distance
        self._min_distance = self._prev_distance
        self._steps = 0
        self._path_length = 0.0
        self._prev_pose = self._robot_pose()
        self._episode = self.vln_sim.current_episode
        self._frames = []
        self._infos = []
        self._rewards = []

        rgb = self._rgb(obs)
        info_out = {
            "episode_label": episode_label,
            "distance_to_goal": self._prev_distance,
            "start_distance": self._start_distance,
        }
        self._infos.append(info_out)
        if not self.minimal_logging:
            self._frames.append(rgb.copy())
        return rgb, {
            "obs": {"instr_or_goal": self._instruction()},
            "reward": 0.0,
            "done": False,
            "is_exhausted": self.is_exhausted(),
            "info": info_out,
        }

    def step(self, action, supplementary_logs: Optional[Dict[str, Any]] = None):
        """`action` is the `(gap, 3)` chunk the policy head already truncated."""
        chunk = np.asarray(action, dtype=np.float64).reshape(-1, 3)
        if len(chunk) != self.gap:
            raise ValueError(
                f"env gap={self.gap} but received a {len(chunk)}-row chunk; the policy head "
                "and this env must agree on ticks-per-step or sim time and policy steps "
                "quietly stop meaning the same thing across runs."
            )
        position, quat = self._robot_pose()
        world_waypoints = _relative_se2_to_world(position, quat, chunk)
        duration_s = len(world_waypoints) * self.dt
        self.vln_sim._handle_timed_trajectory_command(
            {"waypoints": world_waypoints.tolist(), "dt": self.dt}
        )
        obs, native_reward, native_done, info = self._step_until(
            lambda: self.vln_sim.timed_trajectory is None,
            timeout_sim_s=duration_s + self.timeout_margin_s,
            on_timeout=lambda: self.vln_sim._clear_timed_trajectory("rl_env_timeout"),
            capture_video=not self.minimal_logging,
        )
        self._steps += 1

        measurements = (info or {}).get("measurements", info or {})
        distance = measurements.get("distance_to_goal")
        distance = float(distance) if distance is not None else None
        collision = float(measurements.get("collision", 0.0) or 0.0)
        termination_reason = (info or {}).get("terminations", {}).get("termination_reason")

        progress = 0.0
        if distance is not None and self._prev_distance is not None:
            progress = self._prev_distance - distance
            if self.progress_reward_clip > 0:
                progress = float(
                    np.clip(progress, -self.progress_reward_clip, self.progress_reward_clip)
                )
        reward = float(progress) - self.slack_penalty
        if collision:
            reward -= self.collision_penalty

        new_position, new_quat = self._robot_pose()
        self._path_length += float(np.linalg.norm(new_position[:2] - self._prev_pose[0][:2]))
        self._prev_pose = (new_position, new_quat)

        if distance is not None:
            self._prev_distance = distance
            if self._min_distance is None or distance < self._min_distance:
                self._min_distance = distance

        reached = bool(distance is not None and distance <= self.success_distance)
        if reached and self.success_reward:
            reward += self.success_reward

        native_truncated, native_failure = _native_termination_flags(
            termination_reason, native_done=bool(native_done), reached=reached
        )
        if native_failure and self.failure_penalty:
            reward -= self.failure_penalty
        truncated = bool(
            not reached and (self._steps >= self.max_steps or native_truncated)
        )
        done = bool(reached or truncated or native_done)

        oracle_success = bool(self._min_distance is not None and self._min_distance <= self.success_distance)
        start_distance = self._start_distance or 0.0
        oracle_spl = (
            start_distance / max(start_distance, self._path_length)
            if oracle_success and start_distance > 0
            else 0.0
        )
        info_out = {
            "episode_label": self._episode.episode_label if self._episode is not None else None,
            "distance_to_goal": distance,
            "start_distance": self._start_distance,
            "success": reached,
            "oracle_success": oracle_success,
            "oracle_spl": oracle_spl,
            "path_length": self._path_length,
            "collision": collision,
            "distance_progress": progress,
            "truncated": truncated,
            "steps": self._steps,
            "termination_reason": termination_reason,
            "native_failure": native_failure,
            "failure_penalty": self.failure_penalty if native_failure else 0.0,
        }
        self._infos.append(info_out)
        self._rewards.append(reward)

        rgb = self._rgb(obs)
        if not self.minimal_logging and self.video_tick_stride == 0:
            self._frames.append(rgb.copy())
        return rgb, {
            "obs": {"instr_or_goal": self._instruction()},
            "reward": reward,
            "done": done,
            "is_exhausted": self.is_exhausted(),
            "info": info_out,
        }

    def flush_logs_to_disk(self, clear_steps: bool = True):
        """Write this episode's summary/sequence (+ optional MP4) and ship it to the logger.

        Mirrors `objectnav_continuous.ContinuousObjectNavEnvActor.flush_logs_to_disk`'s
        contract -- scalar keys plus `vid/episode_video` / `img/thumbnail` in one payload
        row -- so `logging_workers` wandb-wraps this identically to the habitat-continuous
        runs and the two are comparable on the same charts.
        """
        import json

        if self.logging_output_dir is None or not self._infos:
            if clear_steps:
                self._frames, self._infos, self._rewards = [], [], []
            return None

        label = str(self._infos[0].get("episode_label", f"ep_{time.time():.0f}"))
        save_dir = os.path.join(
            self.logging_output_dir, f"{label}.{os.getpid()}@{time.time():.0f}"
        )
        os.makedirs(save_dir, exist_ok=True)

        last = self._infos[-1]
        episode_logs: Dict[str, Any] = {
            "episode_label": label,
            "n_steps": len(self._rewards),
            "success": int(bool(last.get("success", False))),
            "oracle_success": int(bool(last.get("oracle_success", False))),
            "oracle_spl": float(last.get("oracle_spl", 0.0)),
            "distance_to_goal": last.get("distance_to_goal"),
            "start_distance": last.get("start_distance"),
            "path_length": float(last.get("path_length", 0.0)),
            "truncated": int(bool(last.get("truncated", False))),
            "mean_reward": float(np.mean(self._rewards)) if self._rewards else 0.0,
            "collision_rate": float(np.mean([bool(i.get("collision")) for i in self._infos])),
            "worker_pid": os.getpid(),
            "timestamp": time.time(),
        }
        if not self.minimal_logging and len(self._frames) > 1:
            try:
                import imageio

                video_path = os.path.join(save_dir, "video.mp4")
                fps = self.video_fps
                if self.video_tick_stride > 0:
                    fps = self.video_realtime_factor / (self.dt * self.video_tick_stride)
                imageio.mimsave(video_path, self._frames, fps=fps)
                episode_logs["vid/episode_video"] = video_path
                from PIL import Image

                thumb_path = os.path.join(save_dir, "thumbnail.jpg")
                Image.fromarray(self._frames[-1]).save(thumb_path, quality=85)
                episode_logs["img/thumbnail"] = thumb_path
            except Exception as e:  # video failure must never kill a training episode
                print(f"[navverse_env] video write failed: {e}")

        with open(os.path.join(save_dir, "sequence.json"), "w") as f:
            json.dump(
                {
                    "distance_to_goal": [i.get("distance_to_goal") for i in self._infos],
                    "reward": self._rewards,
                },
                f,
            )
        with open(os.path.join(save_dir, "summary.json"), "w") as f:
            json.dump(episode_logs, f, indent=2)
        with open(os.path.join(self.logging_output_dir, f"results_{os.getpid()}"), "a") as f:
            f.write(json.dumps({k: v for k, v in episode_logs.items() if not k.startswith(("vid/", "img/"))}) + "\n")

        if clear_steps:
            self._frames, self._infos, self._rewards = [], [], []

        if self.logger_actor is not None:
            import ray

            try:
                row = (
                    {k if k.startswith(("vid/", "img/")) else self._log_prefix + k: v
                     for k, v in episode_logs.items()}
                    if self._log_prefix
                    else episode_logs
                )
                ray.get(self.logger_actor.log_row.remote(row=row), timeout=1.0)
            except Exception as e:
                print(f"[navverse_env] logger ack issue: {e}")
        return os.path.join(save_dir, "summary.json")


# =========================================================================================
# Batched variant: NavVerseHostActor + NavVerseBatchCoordinatorActor + slot proxies
# =========================================================================================
#
# `NavVerseEnvActor` above runs vector_envs=1 per Ray actor, matching flowsde's assumption
# that each `sim` handle is an independent single-episode process. On NavVerse that is
# expensive: one Isaac Lab process is ~60 GB VRAM (benchmark_longnav_continuous.yaml's
# sim_vram_gb) as a FIXED cost paid once regardless of how many robots share it, so 1
# actor = 1 robot wastes the 8x amortization the old vectorized discrete env got by running
# `vector_envs=8` inside one process.
#
# This variant recovers that amortization while still presenting flowsde's rollout_core
# with what it expects -- a flat list of independent-looking `sim` handles, each with the
# ordinary five-method interface. `NavVerseHostActor` owns ONE Isaac Lab process running
# `slots_per_host` vectorized robots. A lightweight async
# `NavVerseBatchCoordinatorActor` collects each host's per-slot reset/action calls and
# submits exactly one batch call to the strictly single-threaded Isaac host. One
# `NavVerseSlotProxyActor` per robot slot forwards the ordinary env interface through that
# coordinator. `collect_rollouts` never learns the difference: it just sees
# `num_hosts * slots_per_host` handles whose futures resolve, some faster than others.
#
# THE COST OF THIS TRICK, spelled out because it is easy to misread as free parallelism:
# The coordinator blocks a proxy's caller until ALL CURRENTLY ACTIVE slots on that host
# have submitted their action/reset for the current round --
# because Isaac Lab's vectorized `env.step()` advances every robot in the batch on one
# shared GPU tick; there is no such thing as stepping one of the 8 without the others. A
# slow VLM inference on ANY one of a host's 8 slots therefore stalls all 7 siblings' policy
# steps too, every single step -- the same lockstep coupling the pre-flowsde vectorized env
# had, just now hidden behind an interface that looks per-actor independent. This shows up
# as elevated `sim_latency` in vlm_logs, not a scheduler hang, PROVIDED no slot's step call
# is ever abandoned (e.g. its VLM worker actor crashes mid-episode) -- an abandoned slot
# would stall the whole host's batch forever, since `all(...)` never becomes true.
# The coordinator is deliberately separate from the host: Ray actors configured with
# `max_concurrency>1` run methods on `Dummy-*` worker threads even though `__init__` runs on
# `MainThread`. Isaac Sim must be initialized and driven from the process main thread. The
# earlier in-host threaded barrier therefore initialized successfully and then froze on the
# first vectorized reset. Keep the host single-threaded and put only pure-Python/Ray
# synchronization in the async coordinator.
#
# Episode/scene assignment is fully owned by the host, not by `assign_shard`/the shard
# iterator: every proxy's `assign_shard`/`is_exhausted` are pass-throughs onto shared,
# host-level state (see `_load_pool`/`_build_scene_rounds`, reusing the scene-major
# ordering from `NavVerseEnvActor` above), because a host's `slots_per_host` robots must
# always share ONE scene per round -- exactly the constraint the pre-flowsde vectorized
# `reset_batch` enforced (`scene_paths` must have length 1). `episodes_path` should
# therefore group into rounds of size `slots_per_host` per scene (this repo's smoke sample,
# 16 scenes x 8 episodes, matches `slots_per_host=8` for exactly this reason).


class NavVerseHostActor:
    """Owns one Isaac Lab process running `slots_per_host` vectorized robots.

    Not registered as a Hydra `sim` target directly -- `NavVerseSlotProxyActor` creates
    (or attaches to) one of these per `host_index`, lazily and race-safely, via a Ray
    detached named actor. See the module docstring above for the batching/coupling
    contract this implements.
    """

    def __init__(
        self,
        slots_per_host: int = 8,
        navverse_repo_root: str = "/home/ubuntu/Projects/NavVerse-Benchmark",
        config_path: str = "configs/default.yaml",
        episode_folder: str = "/home/ubuntu/Projects/navverse_data/episodes/",
        scene_folder: str = "/home/ubuntu/Projects/navverse_data/",
        episodes_path: Optional[str] = None,
        train_uids: Optional[Any] = None,
        excluded_episode_labels: Optional[List[str]] = None,
        task_type: str = "placenav",
        robot_name: str = "spot",
        tidybot_embodiment: str = "full",
        hide_robot_visuals: bool = False,
        on_demand_render: bool = False,
        camera_rgb_only: bool = False,
        policy_camera_only: bool = False,
        disable_camera: bool = False,
        trajectory_planner: Optional[str] = None,
        profile_step_timing: bool = False,
        profile_step_interval: int = 100,
        gap: int = 10,
        dt: float = 0.04,
        max_steps: int = 175,
        success_distance: float = 1.6,
        slack_penalty: float = 0.0,
        collision_penalty: float = 0.0,
        failure_penalty: float = 0.0,
        progress_reward_clip: float = 0.75,
        success_reward: float = 0.0,
        timeout_margin_s: float = 1.0,
        minimal_logging: bool = False,
        video_tick_stride: int = 0,
        video_realtime_factor: float = 1.0,
        logging_output_dir: Optional[str] = None,
        logger_actor: Any = None,
        **kwargs: Any,
    ):
        self.n = int(slots_per_host)
        self.gap = int(gap)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.success_distance = float(success_distance)
        self.slack_penalty = float(slack_penalty)
        self.collision_penalty = float(collision_penalty)
        self.failure_penalty = float(failure_penalty)
        self.progress_reward_clip = float(progress_reward_clip)
        self.success_reward = float(success_reward)
        self.timeout_margin_s = float(timeout_margin_s)
        self.minimal_logging = bool(minimal_logging)
        self.policy_camera_only = bool(policy_camera_only)
        self.on_demand_render = bool(on_demand_render)
        self.video_tick_stride = int(video_tick_stride)
        self.video_realtime_factor = float(video_realtime_factor)
        if self.video_tick_stride < 0:
            raise ValueError("video_tick_stride must be non-negative")
        if self.video_realtime_factor <= 0:
            raise ValueError("video_realtime_factor must be positive")
        self.logging_output_dir = logging_output_dir
        self.logger_actor = logger_actor

        self._episodes_path = episodes_path
        self._train_uids = _uid_filter(train_uids)
        self._excluded_episode_labels = set(excluded_episode_labels or [])
        self._pool: Optional[List[str]] = None
        self._scene_rounds: List[List[str]] = []
        self._round_ptr = 0
        self._assigned_round: Optional[List[str]] = None
        self._rng = np.random.default_rng(os.getpid())

        self._prev_distance: List[Optional[float]] = [None] * self.n
        self._start_distance: List[Optional[float]] = [None] * self.n
        self._min_distance: List[Optional[float]] = [None] * self.n
        self._path_length = [0.0] * self.n
        self._prev_pose = [None] * self.n
        self._episode_label: List[Optional[str]] = [None] * self.n
        self._steps = [0] * self.n
        self._done = [True] * self.n  # everyone "done" until the first reset round runs
        self._infos: List[List[Dict[str, Any]]] = [[] for _ in range(self.n)]
        self._rewards: List[List[float]] = [[] for _ in range(self.n)]
        self._actions: List[List[Any]] = [[] for _ in range(self.n)]
        self._frames: List[List[np.ndarray]] = [[] for _ in range(self.n)]

        self._reset_results: Dict[int, Any] = {}
        self._step_results: Dict[int, Any] = {}
        self._vector_results: Dict[int, Any] = {}
        self._isaac_thread_id = threading.get_ident()
        self.manager_profile_stats: Dict[str, tuple] = {}
        self.host_profile_stats: Dict[str, tuple] = {}
        self.manager_profile_hooks_installed = False
        self.profile_step_timing = bool(profile_step_timing)

        self.followers = None  # built once episodes are loaded (needs vln_sim/robot count)
        self._boot_isaac_lab_vectorized(
            navverse_repo_root=navverse_repo_root,
            config_path=config_path,
            episode_folder=episode_folder,
            scene_folder=scene_folder,
            task_type=task_type,
            robot_name=robot_name,
            tidybot_embodiment=tidybot_embodiment,
            hide_robot_visuals=hide_robot_visuals,
            on_demand_render=on_demand_render,
            camera_rgb_only=camera_rgb_only,
            policy_camera_only=policy_camera_only,
            disable_camera=disable_camera,
            trajectory_planner=trajectory_planner,
            profile_step_timing=profile_step_timing,
            profile_step_interval=profile_step_interval,
        )

    def _boot_isaac_lab_vectorized(
        self,
        *,
        navverse_repo_root,
        config_path,
        episode_folder,
        scene_folder,
        task_type,
        robot_name,
        tidybot_embodiment,
        hide_robot_visuals,
        on_demand_render,
        camera_rgb_only,
        policy_camera_only,
        disable_camera,
        trajectory_planner,
        profile_step_timing,
        profile_step_interval,
    ):
        def is_linux_headless():
            has_x11 = "DISPLAY" in os.environ
            has_wayland = "WAYLAND_DISPLAY" in os.environ
            return not (has_x11 or has_wayland)

        # See NavVerseEnvActor._boot_isaac_lab's identical chdir for why: NavVerse resolves
        # its Hydra config/episode/scene paths relative to CWD, not to this actor's own
        # inherited CWD (this RL trainer's directory).
        os.chdir(navverse_repo_root)

        import argparse
        import sys

        from isaaclab.app import AppLauncher
        import navverse.utils.rsl_rl_cli_args as rsl_rl_cli_args
        import navverse.vln_args as vln_cli_args

        parser = argparse.ArgumentParser(description="NavVerse RL host actor")
        rsl_rl_cli_args.add_rsl_rl_args(parser)
        vln_cli_args.add_vln_args(parser)
        AppLauncher.add_app_launcher_args(parser)

        original_argv = sys.argv.copy()
        navverse_args = [
            "--episode_folder", str(episode_folder),
            "--scene_folder", str(scene_folder),
            "--task_type", str(task_type),
            "--robot_name", str(robot_name),
            "--tidybot_embodiment", str(tidybot_embodiment),
            "--timed_trajectory_control_dt", str(self.dt),
            "--on_demand_render", str(bool(on_demand_render)),
            "--disable_camera", str(bool(disable_camera)),
            "--profile_step_timing", str(bool(profile_step_timing)),
            "--profile_step_interval", str(int(profile_step_interval)),
            "--num_envs", str(self.n),
        ]
        if str(config_path).lower() != "auto":
            navverse_args.extend(["--config", str(config_path)])
        if trajectory_planner:
            navverse_args.extend(["--trajectory_planner", str(trajectory_planner)])

        sys.argv = [sys.argv[0], *navverse_args]
        args = vln_cli_args.parse_args(parser)
        sys.argv = original_argv
        args.disable_socket_server = True
        args.headless = is_linux_headless()

        if camera_rgb_only:
            os.environ["NAVVERSE_CAMERA_RGB_ONLY"] = "1"
        os.environ["NAVVERSE_ON_DEMAND_CAMERAS"] = (
            "pov_camera" if policy_camera_only else "pov_camera,third_person_camera"
        )
        app_launcher = AppLauncher(args)
        self.simulation_app = app_launcher.app

        import omni.kit.app

        omni.kit.app.get_app().update()
        from navverse.sim import VLNSim

        self.vln_sim = VLNSim(args)
        self._hide_robot_visuals = bool(hide_robot_visuals)
        # Vectorized rollout must not spend render time on the (unused, off-camera) update
        # hook path -- mirrors the pre-flowsde vectorized env's same override.
        self.vln_sim.update_obs = lambda obs, info: None
        self.device = self.vln_sim.device

    def _install_manager_profile_hooks(self):
        if self.manager_profile_hooks_installed:
            return
        _install_manager_profile_hooks(self.vln_sim, self.manager_profile_stats)
        self.manager_profile_hooks_installed = True

    def get_vector_profile(self, reset=True):
        return _collect_profile_stats(
            self.vln_sim,
            self.manager_profile_stats,
            reset=bool(reset),
            host_stats=self.host_profile_stats,
        )

    def _host_profile_record(self, key: str, started: float) -> None:
        if not self.profile_step_timing:
            return
        total_s, count = self.host_profile_stats.get(key, (0.0, 0))
        self.host_profile_stats[key] = (
            total_s + time.perf_counter() - started,
            count + 1,
        )

    # -- episode pool: scene-major, shared across all `n` slots -----------------------------
    def _load_pool(self) -> None:
        if self._pool is not None:
            return
        if not self._episodes_path:
            raise ValueError(
                "NavVerseHostActor has no episodes_path configured; a batched host cannot "
                "fall back to 'whatever VLNSim happens to have loaded' because slots must "
                "share one scene per round, which requires knowing the scene grouping."
            )
        with open(self._episodes_path) as f:
            self._pool = [line.strip() for line in f if line.strip()]
        missing = self._excluded_episode_labels.difference(self._pool)
        if missing:
            raise ValueError(f"Excluded episode labels are absent from the pool: {sorted(missing)}")
        if self._excluded_episode_labels:
            self._pool = [label for label in self._pool if label not in self._excluded_episode_labels]
            print(f"[navverse_host] excluded {len(self._excluded_episode_labels)} invalid episode(s)")
        if not self._pool:
            raise ValueError(f"episodes_path={self._episodes_path} resolved to zero labels")
        if self._train_uids is not None:
            missing_train = self._train_uids.difference(self._pool)
            if missing_train:
                raise ValueError(
                    f"train_uids contains labels absent from the pool: {sorted(missing_train)}"
                )
            print(
                f"[navverse_host] train_uids: serving {len(self._train_uids)} of "
                f"{len(self._pool)} episodes for training"
            )

    def _build_scene_rounds(self) -> None:
        """One round = `self.n` episodes from ONE scene (Isaac's batched reset needs all
        slots on the same scene). Scenes with fewer than `n` episodes are skipped with a
        loud warning rather than padded with repeats, since a repeat would silently bias
        that scene's sampling within a round.
        """
        labels = (
            [label for label in self._pool if label in self._train_uids]
            if self._train_uids is not None
            else self._pool
        )
        by_scene: Dict[str, List[str]] = {}
        for label in labels:
            by_scene.setdefault(_scene_of(label), []).append(label)
        scenes = [s for s, eps in by_scene.items() if len(eps) >= self.n]
        skipped = set(by_scene) - set(scenes)
        if skipped:
            print(
                f"[navverse_host] {len(skipped)} scene(s) have fewer than "
                f"slots_per_host={self.n} episodes and will be skipped this pass: "
                f"{sorted(skipped)}"
            )
        if not scenes:
            raise ValueError(
                f"No scene in {self._episodes_path} has >= slots_per_host={self.n} episodes"
            )
        self._rng.shuffle(scenes)
        rounds = []
        for scene in scenes:
            eps = list(by_scene[scene])
            self._rng.shuffle(eps)
            for start in range(0, len(eps) - self.n + 1, self.n):
                rounds.append(eps[start : start + self.n])
        self._scene_rounds = rounds
        self._round_ptr = 0

    def _next_round(self) -> List[str]:
        if self._assigned_round is not None:
            labels = self._assigned_round
            self._assigned_round = None
            return labels
        self._load_pool()
        if self._round_ptr >= len(self._scene_rounds):
            self._build_scene_rounds()
        labels = self._scene_rounds[self._round_ptr]
        self._round_ptr += 1
        return labels

    # -- single-threaded batch interface (called by NavVerseBatchCoordinatorActor) ---------
    def _assert_isaac_main_thread(self) -> None:
        if threading.get_ident() != self._isaac_thread_id:
            raise RuntimeError(
                "NavVerseHostActor must remain a default single-threaded Ray actor: "
                "Isaac Sim was initialized on the actor main thread and cannot be driven "
                "from a max_concurrency worker thread"
            )

    def assign_shard_slot(self, slot: int, episodes: Optional[List[str]]) -> None:
        # Trivial-shard convention only (see NavVerseEnvConfig.episodes_path docs): the host
        # owns dataset scope, `episodes` here is only ever None in the flows this port
        # supports (training's shard_size=0), so there is nothing to act on beyond ensuring
        # the pool is loaded.
        self._load_pool()

    def assign_shard(self, episodes: Optional[List[str]] = None) -> None:
        """Assign the next same-scene vector round, matching the legacy batch API."""
        if episodes is None:
            self._load_pool()
            return
        labels = list(episodes)
        if len(labels) != self.n:
            raise ValueError(f"Expected {self.n} episode labels, got {len(labels)}")
        scenes = {_scene_of(label) for label in labels}
        if len(scenes) != 1:
            raise ValueError(
                f"A NavVerse vector round must contain one scene, got {sorted(scenes)}"
            )
        self._assigned_round = labels

    def is_exhausted(self) -> bool:
        return False

    def list_episode_uids(self) -> List[str]:
        self._load_pool()
        return list(self._pool)

    def worker_placement(self) -> Dict[str, Any]:
        return {
            "pid": os.getpid(),
            "ray_gpu_ids": ray.get_gpu_ids(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slots_per_host": self.n,
        }

    def is_exhausted_slot(self, slot: int) -> bool:
        return False  # the host loops its pool forever, matching trivial-shard semantics

    def set_log_prefix_slot(self, slot: int, prefix: str) -> None:
        pass  # not implemented for the batched host in this version

    def set_media_enabled(self, enabled: bool) -> None:
        self.minimal_logging = not bool(enabled)
        if self.policy_camera_only:
            self.vln_sim.set_on_demand_camera_names(
                ("pov_camera", "third_person_camera")
                if enabled
                else ("pov_camera",)
            )
        if self.minimal_logging:
            self._frames = [[] for _ in range(self.n)]

    @torch.inference_mode()
    def reset_batch(self) -> Dict[int, Any]:
        self._assert_isaac_main_thread()
        if not all(self._done):
            active = [i for i, done in enumerate(self._done) if not done]
            raise RuntimeError(f"Cannot reset NavVerse batch while slots are active: {active}")
        self._do_batch_reset()
        self._done = [False] * self.n
        self._vector_results = dict(self._reset_results)
        return self._reset_results

    def reset_vector(self):
        results = self.reset_batch()
        return (
            np.stack([results[index][0] for index in range(self.n)]),
            [results[index][1] for index in range(self.n)],
        )

    def _robot_poses(self):
        root = self.vln_sim.env._pose_state_w().detach().cpu().numpy()
        positions = root[:, :3].astype(np.float64)
        quats_wxyz = root[:, 3:7].astype(np.float64)
        quats_xyzw = np.concatenate([quats_wxyz[:, 1:], quats_wxyz[:, :1]], axis=1)
        return positions, quats_xyzw

    def _sim_time(self) -> float:
        manager_env = getattr(self.vln_sim, "manager_env", None)
        if manager_env is None:
            return 0.0
        return float(getattr(manager_env, "common_step_counter", 0) or 0) * float(
            getattr(manager_env, "step_dt", 0.0) or 0.0
        )

    def _wait_for_scene_ready(self, label: str) -> None:
        wall_start = time.time()
        iterations = 0
        while not _scene_request_is_ready(self.vln_sim, label):
            self.vln_sim.step()
            iterations += 1
            if self.vln_sim.sim_state == "terminated":
                raise RuntimeError(
                    f"NavVerse terminated while loading requested episode {label}"
                )
            elapsed = time.time() - wall_start
            if iterations % 200 == 0:
                print(
                    f"[navverse_host] still waiting for scene ready ({label}): "
                    f"sim_state={self.vln_sim.sim_state}, "
                    f"reset_flag={self.vln_sim.reset_flag}, {iterations} iterations, "
                    f"{elapsed:.1f}s elapsed"
                )
            if elapsed > 180.0:
                raise RuntimeError(
                    f"[navverse_host] scene never reached sim_state='running' for "
                    f"{label} after {elapsed:.1f}s / {iterations} step() calls "
                    f"(sim_state={self.vln_sim.sim_state!r}, "
                    f"reset_flag={self.vln_sim.reset_flag!r}, "
                    f"current_episode={getattr(self.vln_sim.current_episode, 'episode_label', None)!r})"
                )
        print(f"[navverse_host] batch scene ready for {label} after {iterations} steps")

    def _do_batch_reset(self) -> None:
        """Reset all vectorized robots to one same-scene episode round."""
        import copy

        reset_started = time.perf_counter()
        slots = range(self.n)
        labels = self._next_round()
        episode_map = {ep.episode_label: ep for ep in self.vln_sim.episode_list}
        episodes = [episode_map[label] for label in labels]
        scene_paths = {ep["path"] for ep in episodes}
        if len(scene_paths) != 1:
            raise ValueError(
                "A batch round must share one scene; _build_scene_rounds should have "
                f"guaranteed this but got scenes={scene_paths} for labels={labels}"
            )

        scene_started = time.perf_counter()
        scene_request_started = time.perf_counter()
        self.vln_sim.load_episode(labels[0])
        self._host_profile_record("reset/scene_load_request", scene_request_started)
        scene_wait_started = time.perf_counter()
        self._wait_for_scene_ready(labels[0])
        self._host_profile_record("reset/scene_ready_wait", scene_wait_started)
        self._host_profile_record("reset/scene_load", scene_started)
        if self.profile_step_timing:
            self._install_manager_profile_hooks()
        if self._hide_robot_visuals:
            hidden_count = _set_robot_visuals_hidden(self.vln_sim, True)
            print(f"[navverse_host] hidden robot visual prims: {hidden_count}")

        pose_reset_started = time.perf_counter()
        env = self.vln_sim.env
        robot = env.scene["robot"]
        with torch.inference_mode():
            root_state = robot.data.root_state_w.clone()
            for i, ep in enumerate(episodes):
                position = list(ep["start_position"])
                spawn_height = getattr(env.manager_env.cfg, "robot_spawn_height", 0.6)
                if getattr(env.manager_env.cfg, "use_episode_robot_spawn_height", True):
                    spawn_height = ep.get("robot_spawn_height", spawn_height)
                position[2] += float(spawn_height)
                root_state[i, :3] = torch.tensor(position, device=robot.device, dtype=root_state.dtype)
                root_state[i, 3:7] = torch.tensor(ep["start_rotation"], device=robot.device, dtype=root_state.dtype)
                root_state[i, 7:] = 0.0
            robot.write_root_state_to_sim(root_state)
            if env.control_mode == "physics_velocity":
                joint_pos = torch.zeros_like(robot.data.joint_pos)
                joint_vel = torch.zeros_like(robot.data.joint_vel)
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.set_joint_velocity_target(joint_vel)
            robot.write_data_to_sim()
            env._reset_low_level_policy_state()
            raw_obs = None
            for _ in range(2):
                if self.vln_sim.robot_name == "tidybot":
                    low_level_action = env._empty_env_action()
                else:
                    zero_commands = torch.zeros(
                        (self.n, self.vln_sim.command_dim),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    low_level_action = env._command_to_env_action(zero_commands)
                raw_obs, _, _, _ = env._step_low_level_env(low_level_action)
                env.obs = raw_obs
            obs = env._get_high_level_obs(raw_obs)
        self._host_profile_record("reset/pose_and_settle", pose_reset_started)
        print(f"[navverse_host] reset {self.n} vector robot poses")

        state_started = time.perf_counter()
        follower = (
            self.vln_sim.tidybot_timed_trajectory_follower
            if self.vln_sim.robot_name == "tidybot"
            else self.vln_sim.timed_trajectory_follower
        )
        self.followers = [copy.deepcopy(follower) for _ in range(self.n)]

        positions, _ = self._robot_poses()
        for i, ep in enumerate(episodes):
            distance, _ = NavVerseEnvActor._distance_to_goal(ep, positions[i])
            self._episode_label[i] = ep.episode_label
            self._prev_distance[i] = float(distance)
            self._start_distance[i] = float(distance)
            self._min_distance[i] = float(distance)
            self._path_length[i] = 0.0
            self._steps[i] = 0
            self._infos[i] = []
            self._rewards[i] = []
            self._actions[i] = []
            self._frames[i] = []
        _, quats = self._robot_poses()
        for i in range(self.n):
            self._prev_pose[i] = (positions[i].copy(), quats[i].copy())
        self._host_profile_record("reset/episode_state", state_started)

        render_started = time.perf_counter()
        if self.on_demand_render:
            rgb_batch, third_person_batch = self._render_rgb_batches(
                include_third_person=not self.minimal_logging
            )
        else:
            rgb_batch = self._rgb_batch(obs)
            third_person_batch = None
        self._host_profile_record("reset/render", render_started)
        self._reset_results = {}
        for i in slots:
            info_out = {
                "episode_label": self._episode_label[i],
                "distance_to_goal": self._prev_distance[i],
                "start_distance": self._start_distance[i],
            }
            self._infos[i].append(info_out)
            if not self.minimal_logging:
                self._frames[i].append(
                    self._video_frame(rgb_batch, third_person_batch, i)
                )
            self._reset_results[i] = (
                rgb_batch[i],
                {
                    "obs": {"instr_or_goal": self._instruction(episodes[i])},
                    "reward": 0.0,
                    "done": False,
                    "is_exhausted": False,
                    "info": info_out,
                },
            )
        self._host_profile_record("reset/total", reset_started)

    @staticmethod
    def _rgb_batch(obs) -> np.ndarray:
        pov_rgb = NavVerseEnvActor._find_obs_value(obs, "pov_rgb")
        if pov_rgb is None:
            raise KeyError("pov_rgb not found in NavVerse obs")
        return pov_rgb.detach().cpu().numpy()[..., :3].astype(np.uint8)

    def _render_rgb_batches(
        self, *, include_third_person: bool
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        camera_names = ["pov_camera"]
        if include_third_person:
            camera_names.append("third_person_camera")
        self.vln_sim.update_on_demand_sensor_data(camera_names=camera_names)
        sensors = self.vln_sim.manager_env.scene.sensors

        def rgb(camera_name: str) -> np.ndarray:
            output = sensors[camera_name].data.output
            if "rgb" not in output:
                raise KeyError(f"{camera_name} has no RGB output")
            return output["rgb"].detach().cpu().numpy()[..., :3].astype(np.uint8)

        pov_batch = rgb("pov_camera")
        third_person_batch = rgb("third_person_camera") if include_third_person else None
        if len(pov_batch) != self.n:
            raise RuntimeError(
                f"On-demand POV render returned {len(pov_batch)} frames for {self.n} slots"
            )
        return pov_batch, third_person_batch

    @staticmethod
    def _video_frame(pov_batch, third_person_batch, slot: int) -> np.ndarray:
        pov = pov_batch[slot].copy()
        if third_person_batch is None:
            return pov
        third_person = third_person_batch[slot]
        if third_person.shape[:2] != pov.shape[:2]:
            raise ValueError(
                "POV and third-person frames must have matching height and width; "
                f"got {pov.shape} and {third_person.shape}"
            )
        return np.concatenate([pov, third_person], axis=1)

    @staticmethod
    def _instruction(episode) -> str:
        return episode.get("placenav_goal") or episode.get("instruction") or episode.get("objnav") or ""

    @torch.inference_mode()
    def step_batch(self, actions: Dict[int, Any]) -> Dict[int, Any]:
        self._assert_isaac_main_thread()
        active = [i for i in range(self.n) if not self._done[i]]
        if set(actions) != set(active):
            raise ValueError(
                f"Batch action slots must match active slots; got {sorted(actions)}, "
                f"expected {active}"
            )
        self._do_batch_step(active, actions)
        self._vector_results.update(self._step_results)
        return self._step_results

    def step_vector(self, actions: List[Any]):
        if len(actions) != self.n:
            raise ValueError(f"Expected {self.n} vector actions, got {len(actions)}")
        active = [index for index in range(self.n) if not self._done[index]]
        if not active:
            return (
                np.stack([self._vector_results[index][0] for index in range(self.n)]),
                [self._vector_results[index][1] for index in range(self.n)],
            )
        missing = [index for index in active if actions[index] is None]
        if missing:
            raise ValueError(f"Active vector slots have no action: {missing}")
        self.step_batch({index: actions[index] for index in active})
        return (
            np.stack([self._vector_results[index][0] for index in range(self.n)]),
            [self._vector_results[index][1] for index in range(self.n)],
        )

    def force_stop_vector(self, reason: str = "wall_timeout_soft_stop"):
        """Externally truncate every active slot without fabricating a policy action.

        This method is intentionally serviced by the same single-threaded Isaac actor.
        If Isaac is wedged inside reset/step, the request remains queued and the driver
        escalates to process replacement at the hard deadline.
        """
        self._assert_isaac_main_thread()
        if len(self._vector_results) != self.n:
            raise RuntimeError("Cannot force-stop a NavVerse batch before reset completes")
        for slot in range(self.n):
            if self._done[slot]:
                continue
            rgb, state = self._vector_results[slot]
            info = dict(state.get("info", {}))
            oracle_success = bool(
                self._min_distance[slot] is not None
                and self._min_distance[slot] <= self.success_distance
            )
            start_distance = self._start_distance[slot] or 0.0
            info.update(
                {
                    "episode_label": self._episode_label[slot],
                    "distance_to_goal": self._prev_distance[slot],
                    "start_distance": self._start_distance[slot],
                    "success": False,
                    "oracle_success": oracle_success,
                    "oracle_spl": (
                        start_distance / max(start_distance, self._path_length[slot])
                        if oracle_success and start_distance > 0
                        else 0.0
                    ),
                    "path_length": self._path_length[slot],
                    "truncated": True,
                    "steps": self._steps[slot],
                    "termination_reason": str(reason),
                    "external_timeout_stop": True,
                }
            )
            terminal_state = dict(state)
            terminal_state.update(
                {"reward": 0.0, "done": True, "is_exhausted": False, "info": info}
            )
            self._done[slot] = True
            self._infos[slot].append(info)
            self._vector_results[slot] = (rgb, terminal_state)
        return (
            np.stack([self._vector_results[index][0] for index in range(self.n)]),
            [self._vector_results[index][1] for index in range(self.n)],
        )

    def _do_batch_step(self, active_slots: List[int], actions: Dict[int, Any]) -> None:
        step_started = time.perf_counter()
        prepare_started = time.perf_counter()
        world_targets: Dict[int, np.ndarray] = {}
        positions, quats = self._robot_poses()
        for slot, action in actions.items():
            chunk = np.asarray(action, dtype=np.float64).reshape(-1, 3)
            if len(chunk) != self.gap:
                raise ValueError(
                    f"host gap={self.gap} but slot {slot} submitted a {len(chunk)}-row chunk"
                )
            world_targets[slot] = _relative_se2_to_world(positions[slot], quats[slot], chunk)
            self._actions[slot].append(chunk.tolist())
            self.followers[slot].reset()
        self._host_profile_record("step/prepare_targets", prepare_started)

        env = self.vln_sim.env
        raw_reward = np.zeros(self.n, dtype=np.float32)
        termination_reasons: Dict[int, Optional[str]] = {i: None for i in active_slots}
        physical_collisions = np.zeros(self.n, dtype=bool)
        physical_contact_force_max = np.zeros(self.n, dtype=np.float32)
        raw_obs = env.obs
        video_tick = 0
        for tick_index in range(self.gap):
            poses_started = time.perf_counter()
            positions, quats = self._robot_poses()
            self._host_profile_record("tick/read_robot_poses", poses_started)
            command_started = time.perf_counter()
            command_batch = self._command_batch(
                world_targets, positions, quats, tick_index
            )
            self._host_profile_record("tick/waypoint_followers", command_started)
            if self.vln_sim.robot_name == "tidybot":
                if env.control_mode == "physics_velocity":
                    low_level_action = env._command_to_env_action(command_batch)
                else:
                    apply_started = time.perf_counter()
                    self._apply_tidybot_transform_batch(
                        command_batch, active_slots, positions
                    )
                    self._host_profile_record(
                        "tick/apply_tidybot_transform", apply_started
                    )
                    low_level_action = env._empty_env_action()
            else:
                low_level_action = env._command_to_env_action(command_batch)
            env_step_started = time.perf_counter()
            raw_obs, reward, _, _ = env._step_low_level_env(low_level_action)
            self._host_profile_record("tick/low_level_env_step", env_step_started)
            env.obs = raw_obs
            if env.control_mode == "physics_velocity":
                contact_flags, contact_force_max = self._tidybot_contact_state()
                physical_collisions |= contact_flags
                physical_contact_force_max = np.maximum(
                    physical_contact_force_max, contact_force_max
                )
            video_tick += 1
            video_started = time.perf_counter()
            self._capture_batch_video_tick(raw_obs, active_slots, video_tick)
            self._host_profile_record("tick/video_capture", video_started)
            if torch.is_tensor(reward):
                reward_arr = reward.detach().float().cpu().numpy().reshape(-1)
                raw_reward[: len(reward_arr)] += reward_arr[: self.n]
            termination_started = time.perf_counter()
            reset_buf = env.termination_manager.compute()
            done_arr = reset_buf.detach().cpu().numpy().reshape(-1)
            for i in active_slots:
                if termination_reasons[i] is not None:
                    continue
                if len(done_arr) > i and bool(done_arr[i]):
                    terms = env.termination_manager.get_active_iterable_terms(i)
                    termination_reasons[i] = max(terms, key=lambda item: item[1])[0] if terms else "native_termination"
            self._host_profile_record("tick/termination", termination_started)
        env.last_physics_collision = physical_collisions
        env.last_physics_contact_force = physical_contact_force_max
        render_started = time.perf_counter()
        if self.on_demand_render:
            rgb_batch, third_person_batch = self._render_rgb_batches(
                include_third_person=not self.minimal_logging
            )
        else:
            obs = env._get_high_level_obs(raw_obs)
            rgb_batch = self._rgb_batch(obs)
            third_person_batch = None
        self._host_profile_record("step/render", render_started)
        if not self.minimal_logging:
            for slot in active_slots:
                self._frames[slot].append(
                    self._video_frame(rgb_batch, third_person_batch, slot)
                )
        resolve_started = time.perf_counter()
        self._pack_and_resolve(active_slots, rgb_batch, termination_reasons)
        self._host_profile_record("step/reward_and_resolve", resolve_started)
        self._host_profile_record("step/total", step_started)

    def _capture_batch_video_tick(self, raw_obs, active_slots, video_tick) -> None:
        if (
            self.minimal_logging
            or self.on_demand_render
            or self.video_tick_stride <= 0
            or video_tick % self.video_tick_stride
        ):
            return
        tick_obs = self.vln_sim.env._get_high_level_obs(raw_obs)
        tick_rgb = self._rgb_batch(tick_obs)
        for slot in active_slots:
            self._frames[slot].append(tick_rgb[slot].copy())

    def _command_batch(self, world_targets, positions, quats, tick_index):
        commands = []
        tidybot_physics = (
            self.vln_sim.robot_name == "tidybot"
            and self.vln_sim.env.control_mode == "physics_velocity"
        )
        for slot in range(self.n):
            if slot not in world_targets:
                if tidybot_physics:
                    commands.append(np.zeros(3, dtype=np.float64))
                elif self.vln_sim.robot_name == "tidybot":
                    commands.append(
                        np.concatenate([positions[slot], quats[slot]])
                    )
                else:
                    commands.append(torch.zeros((1, 3), device=self.device))
                continue
            waypoints = world_targets[slot]
            target = waypoints[min(tick_index, len(waypoints) - 1)]
            if self.vln_sim.robot_name == "tidybot":
                current_yaw = _yaw_from_xyzw(quats[slot])
                next_pose = self.followers[slot].update(
                    np.array(
                        [positions[slot, 0], positions[slot, 1], current_yaw]
                    ),
                    target,
                    self.dt,
                )
                if tidybot_physics:
                    commands.append(self.followers[slot].previous_command.copy())
                    continue
                half_yaw = 0.5 * float(next_pose[2])
                commands.append(
                    np.array(
                        [
                            next_pose[0],
                            next_pose[1],
                            positions[slot, 2],
                            0.0,
                            0.0,
                            math.sin(half_yaw),
                            math.cos(half_yaw),
                        ],
                        dtype=np.float64,
                    )
                )
            else:
                commands.append(
                    self.followers[slot].update(
                        positions[slot], quats[slot], [target], verbose=False
                    )
                )
        if self.vln_sim.robot_name == "tidybot":
            return np.stack(commands)
        return torch.cat(commands, dim=0)

    def _tidybot_contact_state(self) -> tuple[np.ndarray, np.ndarray]:
        sensor = self.vln_sim.env.scene.sensors.get("contact_forces")
        if sensor is None or sensor.data.net_forces_w is None:
            zeros = np.zeros(self.n, dtype=np.float32)
            return zeros.astype(bool), zeros
        forces = sensor.data.net_forces_w_history
        if forces is None:
            forces = sensor.data.net_forces_w
        planar_force_norm = torch.linalg.vector_norm(
            forces.detach()[..., :2], dim=-1
        )
        threshold = float(getattr(sensor.cfg, "force_threshold", 1.0))
        reduce_dims = tuple(range(1, planar_force_norm.ndim))
        force_max = (
            planar_force_norm.amax(dim=reduce_dims).cpu().numpy().astype(np.float32)
        )
        return force_max > threshold, force_max

    def _apply_tidybot_transform_batch(
        self, transforms, active_slots, current_positions
    ):
        total_started = time.perf_counter()
        env = self.vln_sim.env
        robot = env.scene["robot"]
        source_started = time.perf_counter()
        root_pose = robot.data.root_state_w[:, :7].detach().clone()
        self._host_profile_record("tidybot/read_root_pose_gpu", source_started)
        results = []
        accepted_positions = []
        accepted_quaternions = []
        for slot in active_slots:
            slot_started = time.perf_counter()
            target = np.asarray(transforms[slot], dtype=np.float32)
            start = np.asarray(current_positions[slot], dtype=np.float32).copy()
            candidate = target[:3].copy()
            if env.navmesh_interface is not None:
                height_started = time.perf_counter()
                candidate[2] = env._robot_root_z_for_navmesh_xy(candidate[:2], start[2])
                self._host_profile_record("tidybot/navmesh_height", height_started)
                clearance_started = time.perf_counter()
                if not env._is_navmesh_point_clear(start):
                    start = env._nearest_navmesh_robot_root_pose(start)
                self._host_profile_record("tidybot/navmesh_start_clearance", clearance_started)
            accepted = candidate
            blocked = False
            pose_valid = True
            if env.navmesh_interface is not None:
                valid_started = time.perf_counter()
                pose_valid = env._is_navmesh_pose_valid(start, candidate)
                self._host_profile_record("tidybot/navmesh_pose_valid", valid_started)
            if not pose_valid:
                blocked = True
                if env.collision_response == "block_or_slide":
                    slide_started = time.perf_counter()
                    accepted = env._slide_target_on_navmesh(start, candidate)
                    self._host_profile_record("tidybot/navmesh_slide", slide_started)
                else:
                    accepted = start
                height_started = time.perf_counter()
                accepted[2] = env._robot_root_z_for_navmesh_xy(
                    accepted[:2], accepted[2]
                )
                self._host_profile_record("tidybot/navmesh_blocked_height", height_started)
            accepted_positions.append(accepted)
            accepted_quaternions.append(
                [target[6], target[3], target[4], target[5]]
            )
            results.append(
                {
                    "slot": int(slot),
                    "blocked": bool(blocked),
                    "target": candidate.tolist(),
                    "accepted": accepted.tolist(),
                }
            )
            self._host_profile_record("tidybot/slot_total", slot_started)
        active_indexes = torch.as_tensor(
            active_slots, device=robot.device, dtype=torch.long
        )
        root_pose[active_indexes, :3] = torch.as_tensor(
            np.stack(accepted_positions), device=robot.device, dtype=root_pose.dtype
        )
        root_pose[active_indexes, 3:7] = torch.as_tensor(
            np.asarray(accepted_quaternions),
            device=robot.device,
            dtype=root_pose.dtype,
        )
        write_started = time.perf_counter()
        with torch.inference_mode():
            env._write_tidybot_base_root_pose(root_pose)
            robot.write_data_to_sim()
            env._remember_tidybot_base_pose(root_pose)
        self._host_profile_record("tidybot/write_root_pose", write_started)
        env.last_kinematic_result = results
        self._host_profile_record("tidybot/total", total_started)

    def _pack_and_resolve(self, active_slots, rgb_batch, termination_reasons) -> None:
        """Pack the per-slot results produced by one vectorized host step."""
        self._step_results = {}
        positions, quats = self._robot_poses()
        kinematic_results = {
            int(result["slot"]): result
            for result in getattr(self.vln_sim.env, "last_kinematic_result", [])
            if isinstance(result, dict) and "slot" in result
        }
        physics_collisions = np.asarray(
            getattr(self.vln_sim.env, "last_physics_collision", []), dtype=bool
        )
        physics_contact_forces = np.asarray(
            getattr(self.vln_sim.env, "last_physics_contact_force", []),
            dtype=np.float32,
        )
        for slot in active_slots:
            episode = next(ep for ep in self.vln_sim.episode_list if ep.episode_label == self._episode_label[slot])
            distance, _ = NavVerseEnvActor._distance_to_goal(episode, positions[slot])
            distance = float(distance)
            if self.vln_sim.env.control_mode == "physics_velocity":
                collision = float(
                    len(physics_collisions) > slot and physics_collisions[slot]
                )
            else:
                collision = float(
                    bool(kinematic_results.get(int(slot), {}).get("blocked", False))
                )

            progress = 0.0
            if self._prev_distance[slot] is not None:
                progress = self._prev_distance[slot] - distance
                if self.progress_reward_clip > 0:
                    progress = float(np.clip(progress, -self.progress_reward_clip, self.progress_reward_clip))
            reward = float(progress) - self.slack_penalty

            new_position = positions[slot]
            prev_position, _ = self._prev_pose[slot]
            self._path_length[slot] += float(np.linalg.norm(new_position[:2] - prev_position[:2]))
            self._prev_pose[slot] = (new_position.copy(), self._prev_pose[slot][1])

            self._prev_distance[slot] = distance
            if self._min_distance[slot] is None or distance < self._min_distance[slot]:
                self._min_distance[slot] = distance

            reached = bool(distance <= self.success_distance)
            if reached and self.success_reward:
                reward += self.success_reward
            self._steps[slot] += 1
            termination_reason = termination_reasons.get(slot)
            native_done = termination_reason is not None
            native_truncated, native_failure = _native_termination_flags(
                termination_reason, native_done=native_done, reached=reached
            )
            if native_failure and self.failure_penalty:
                reward -= self.failure_penalty
            truncated = bool(
                not reached and (self._steps[slot] >= self.max_steps or native_truncated)
            )
            done = bool(reached or truncated or native_done)
            self._done[slot] = done

            oracle_success = bool(self._min_distance[slot] <= self.success_distance)
            start_distance = self._start_distance[slot] or 0.0
            oracle_spl = (
                start_distance / max(start_distance, self._path_length[slot])
                if oracle_success and start_distance > 0
                else 0.0
            )
            info_out = {
                "episode_label": self._episode_label[slot],
                "distance_to_goal": distance,
                "start_distance": self._start_distance[slot],
                "success": reached,
                "oracle_success": oracle_success,
                "oracle_spl": oracle_spl,
                "path_length": self._path_length[slot],
                "collision": collision,
                "physics_contact_force": float(
                    physics_contact_forces[slot]
                    if len(physics_contact_forces) > slot
                    else 0.0
                ),
                "distance_progress": progress,
                "truncated": truncated,
                "steps": self._steps[slot],
                "termination_reason": termination_reason,
                "native_failure": native_failure,
                "failure_penalty": self.failure_penalty if native_failure else 0.0,
                "root_z": float(positions[slot, 2]),
                "root_x": float(positions[slot, 0]),
                "root_y": float(positions[slot, 1]),
                "tilt_deg": float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                1.0
                                - 2.0
                                * (quats[slot, 0] ** 2 + quats[slot, 1] ** 2),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                ),
            }
            self._infos[slot].append(info_out)
            self._rewards[slot].append(reward)

            self._step_results[slot] = (
                rgb_batch[slot],
                {
                    "obs": {"instr_or_goal": self._instruction_for_slot(slot)},
                    "reward": reward,
                    "done": done,
                    "is_exhausted": False,
                    "info": info_out,
                },
            )

    def _instruction_for_slot(self, slot: int) -> str:
        episode = next(
            (ep for ep in self.vln_sim.episode_list if ep.episode_label == self._episode_label[slot]),
            None,
        )
        return self._instruction(episode) if episode is not None else ""

    def flush_logs_to_disk_slot(self, slot: int, clear_steps: bool = True):
        import json

        infos, rewards = self._infos[slot], self._rewards[slot]
        if self.logging_output_dir is None or not infos:
            if clear_steps:
                self._infos[slot], self._rewards[slot] = [], []
            return None
        label = str(infos[0].get("episode_label", f"ep_{time.time():.0f}"))
        save_dir = os.path.join(self.logging_output_dir, f"{label}.slot{slot}.{os.getpid()}@{time.time():.0f}")
        os.makedirs(save_dir, exist_ok=True)
        last = infos[-1]
        episode_logs = {
            "episode_label": label,
            "n_steps": len(rewards),
            "success": int(bool(last.get("success", False))),
            "oracle_success": int(bool(last.get("oracle_success", False))),
            "oracle_spl": float(last.get("oracle_spl", 0.0)),
            "distance_to_goal": last.get("distance_to_goal"),
            "start_distance": last.get("start_distance"),
            "path_length": float(last.get("path_length", 0.0)),
            "truncated": int(bool(last.get("truncated", False))),
            "termination_reason": last.get("termination_reason"),
            "root_z": last.get("root_z"),
            "tilt_deg": last.get("tilt_deg"),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "worker_pid": os.getpid(),
            "slot": slot,
            "timestamp": time.time(),
        }
        frames = self._frames[slot]
        if not self.minimal_logging and len(frames) > 1:
            try:
                import imageio
                from PIL import Image

                video_path = os.path.join(save_dir, "video.mp4")
                stride = self.gap if self.on_demand_render else max(self.video_tick_stride, 1)
                fps = self.video_realtime_factor / (self.dt * stride)
                imageio.mimsave(video_path, frames, fps=fps)
                episode_logs["vid/episode_video"] = video_path
                thumbnail_path = os.path.join(save_dir, "thumbnail.jpg")
                Image.fromarray(frames[-1]).save(thumbnail_path, quality=85)
                episode_logs["img/thumbnail"] = thumbnail_path
            except Exception as exc:
                print(f"[navverse_host] video write failed for {label}: {exc}")
        with open(os.path.join(save_dir, "sequence.json"), "w") as f:
            json.dump(
                {
                    "distance_to_goal": [info.get("distance_to_goal") for info in infos],
                    "reward": rewards,
                    "termination_reason": [info.get("termination_reason") for info in infos],
                    "root_xyz": [
                        [info.get("root_x"), info.get("root_y"), info.get("root_z")]
                        for info in infos
                    ],
                    "tilt_deg": [info.get("tilt_deg") for info in infos],
                    "actions": self._actions[slot],
                },
                f,
            )
        with open(os.path.join(save_dir, "summary.json"), "w") as f:
            json.dump(episode_logs, f, indent=2)
        with open(os.path.join(self.logging_output_dir, f"results_{os.getpid()}"), "a") as f:
            f.write(json.dumps(episode_logs) + "\n")
        if clear_steps:
            self._infos[slot], self._rewards[slot] = [], []
            self._actions[slot], self._frames[slot] = [], []
        if self.logger_actor is not None:
            try:
                ray.get(self.logger_actor.log_row.remote(row=episode_logs), timeout=1.0)
            except Exception as e:
                print(f"[navverse_host] logger ack issue: {e}")
        return os.path.join(save_dir, "summary.json")

    def flush_logs_to_disk(self, clear_steps: bool = True):
        return [
            self.flush_logs_to_disk_slot(slot, clear_steps=clear_steps)
            for slot in range(self.n)
        ]


def _get_or_create_named_actor(name: str, actor_cls, options: dict, ctor_kwargs: dict):
    """Race-safe "get or create" for a detached singleton, so `slots_per_host` identically-
    constructed proxies can all reach for the same host without one of them "winning" by
    convention -- Ray's `get_if_exists=True` guarantees exactly one constructor call wins
    and everyone else attaches to that instance.
    """
    return (
        ray.remote(actor_cls)
        .options(name=name, get_if_exists=True, lifetime="detached", **options)
        .remote(**ctor_kwargs)
    )


class NavVerseSlotProxyActor:
    """One robot slot of a vectorized host, presented as an ordinary single-episode env.

    This is the Hydra `_target_` for `sim=navverse_batched`. Each proxy attaches to the
    host's async coordinator for barriered reset/step calls and directly to the
    single-threaded host for non-barriered metadata/logging calls.
    """

    def __init__(
        self,
        num_hosts: int = 8,
        slots_per_host: int = 8,
        host_num_gpus: float = 0.5,
        host_num_cpus: int = 4,
        host_conda_env: Optional[str] = None,
        logging_output_dir: Optional[str] = None,
        logger_actor: Any = None,
        **navverse_kwargs: Any,
    ):
        self._num_hosts = int(num_hosts)
        self._slots_per_host = int(slots_per_host)
        self._navverse_kwargs = dict(navverse_kwargs)
        self._navverse_kwargs["slots_per_host"] = self._slots_per_host
        self._navverse_kwargs["logging_output_dir"] = logging_output_dir
        self._navverse_kwargs["logger_actor"] = logger_actor
        self._host_num_gpus = float(host_num_gpus)
        self._host_num_cpus = int(host_num_cpus)
        self._host_conda_env = host_conda_env

        allocator = _get_or_create_named_actor(
            "navverse_slot_allocator", _SlotAllocator, {"num_cpus": 0}, {}
        )
        self._global_slot = ray.get(allocator.next_slot.remote())
        self._host_index = self._global_slot // self._slots_per_host
        self._local_slot = self._global_slot % self._slots_per_host
        if self._host_index >= self._num_hosts:
            raise ValueError(
                f"resources.num_sims must equal num_hosts*slots_per_host "
                f"({self._num_hosts}*{self._slots_per_host}={self._num_hosts * self._slots_per_host}); "
                f"got a slot index {self._global_slot} that overflows it"
            )

        env_dict = {"conda": self._host_conda_env} if self._host_conda_env else {}
        self._host = _get_or_create_named_actor(
            f"navverse_host_{self._host_index}",
            NavVerseHostActor,
            {
                "num_gpus": self._host_num_gpus,
                "num_cpus": self._host_num_cpus,
                "runtime_env": env_dict,
                "max_restarts": 0,
                "max_task_retries": -1,
            },
            self._navverse_kwargs,
        )
        self._coordinator = _get_or_create_named_actor(
            f"navverse_batch_coordinator_{self._host_index}",
            NavVerseBatchCoordinatorActor,
            {"num_cpus": 0},
            {"host": self._host, "slots_per_host": self._slots_per_host},
        )

    def set_log_prefix(self, prefix: str) -> None:
        ray.get(self._host.set_log_prefix_slot.remote(self._local_slot, prefix))

    def assign_shard(self, episodes: Optional[List[str]] = None) -> None:
        ray.get(self._host.assign_shard_slot.remote(self._local_slot, episodes))

    def is_exhausted(self) -> bool:
        return ray.get(self._host.is_exhausted_slot.remote(self._local_slot))

    def reset(self):
        return ray.get(self._coordinator.reset_slot.remote(self._local_slot))

    def step(self, action, supplementary_logs: Optional[Dict[str, Any]] = None):
        return ray.get(self._coordinator.step_slot.remote(self._local_slot, action))

    def flush_logs_to_disk(self, clear_steps: bool = True):
        return ray.get(self._host.flush_logs_to_disk_slot.remote(self._local_slot, clear_steps))


class NavVerseBatchCoordinatorActor:
    """Async per-host barrier that never imports or drives Isaac Sim.

    Ray may run async-actor code on an event-loop thread; that is safe here because all
    simulator work is submitted as one call to the separate, default single-threaded host.
    """

    def __init__(self, host, slots_per_host: int):
        self._host = host
        self._n = int(slots_per_host)
        self._active_slots = set()
        self._reset_waiters: Dict[int, asyncio.Future] = {}
        self._step_waiters: Dict[int, asyncio.Future] = {}
        self._pending_actions: Dict[int, Any] = {}

    @staticmethod
    def _finish_waiters(waiters: Dict[int, asyncio.Future], results=None, error=None):
        for slot, future in waiters.items():
            if future.done():
                continue
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(results[slot])

    async def reset_slot(self, slot: int):
        if slot in self._reset_waiters:
            raise RuntimeError(f"Slot {slot} submitted reset twice in the same batch round")
        future = asyncio.get_running_loop().create_future()
        self._reset_waiters[slot] = future
        if len(self._reset_waiters) == self._n:
            waiters = self._reset_waiters
            self._reset_waiters = {}
            try:
                results = await self._host.reset_batch.remote()
            except Exception as exc:
                self._finish_waiters(waiters, error=exc)
            else:
                self._active_slots = set(range(self._n))
                self._finish_waiters(waiters, results=results)
        return await future

    async def step_slot(self, slot: int, action):
        if slot not in self._active_slots:
            raise RuntimeError(
                f"Inactive slot {slot} submitted an action; active={sorted(self._active_slots)}"
            )
        if slot in self._step_waiters:
            raise RuntimeError(f"Slot {slot} submitted two actions in the same batch step")
        future = asyncio.get_running_loop().create_future()
        self._step_waiters[slot] = future
        self._pending_actions[slot] = action
        if set(self._step_waiters) == self._active_slots:
            waiters = self._step_waiters
            actions = self._pending_actions
            self._step_waiters = {}
            self._pending_actions = {}
            try:
                results = await self._host.step_batch.remote(actions)
            except Exception as exc:
                self._finish_waiters(waiters, error=exc)
            else:
                self._active_slots = {
                    active_slot
                    for active_slot in self._active_slots
                    if not bool(results[active_slot][1]["done"])
                }
                self._finish_waiters(waiters, results=results)
        return await future


class _SlotAllocator:
    """Hands out sequential global slot indices [0, num_hosts*slots_per_host) so identically-
    constructed proxies can deterministically (if arbitrarily) partition themselves into
    host groups without any of them knowing its own ordinal in advance.
    """

    def __init__(self):
        self._next = 0

    def next_slot(self) -> int:
        slot = self._next
        self._next += 1
        return slot
