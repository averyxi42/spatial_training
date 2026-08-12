"""Continuous ObjectNav as a Ray env actor: one policy step = one action chunk.

Design: `docs/LATENT_RL_ENV.md`. Runs under `habitat_conda_env` (`vln`), which has
`habitat_sim`, `habitat`, `continuous_demos` and `objectnav_eval` importable.

BUILT ON THE INSTALLED `objectnav_eval`, NOT ON A COPY. The forked `longnav/env/sim.py` was
deleted for exactly this reason. Episode sourcing, screening with reason codes, per-dataset
navmesh filename resolution, the live-height geodesic adapter, the PID chunk executor and the
task metrics all live in that package because each of them, done wrong, yields a plausible
wrong number rather than an error -- and the reward depends on every one of them. "Geodesic
progress" means nothing unless it is measured on the robot's navmesh, at the robot's live
height, against a goal on a reachable island.

ONE ENV STEP IS ONE CHUNK, AND ONLY ITS PREFIX RUNS. `ChunkExecutor.execute` tracks the first
`gap` setpoints, one control tick each, and discards the tail -- the same truncation training
and the eval harness use, because the remainder is superseded by the next observation.
Inference is synchronous and assumed instantaneous: no latency masking, no prefix
conditioning. Both are real future work and neither is stubbed, because a stub that silently
does nothing is worse than an absence.

REWARD: geodesic progress per step, with NO TERMINAL TERM. The policy has no stop head, so a
success bonus would optimise the termination heuristic rather than navigation. Termination
belongs to the env: goal reached, step budget spent, or the robot left the world.

SCREENED EPISODES ARE SKIPPED, NOT FAILED. The robot navmesh (r=0.28, h=1.0) is strictly more
fragmented than the one the episodes were cut on, so a goal can sit on an island the robot
cannot reach. Scoring those as zero-reward episodes would teach the policy that some goals are
simply impossible; they are counted and stepped over instead.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class ContinuousObjectNavEnvActor:
    """The five-method env-actor interface, over a physically-simulated robot.

    `reset()` / `step(action)` return `(rgb, state_dict)` with the `obs` / `reward` / `done` /
    `is_exhausted` / `info` keys `rollout_core` consumes.
    """

    def __init__(
        self,
        episodes: str,
        scene_root: Optional[str] = None,
        gap: int = 10,
        dt: float = 0.04,
        max_steps: int = 175,
        success_distance: float = 1.0,
        navmesh: str = "dataset",
        distance_to: str = "VIEW_POINTS",
        sensor_uuid: str = "color_sensor",
        width: int = 640,
        height: int = 480,
        pid_preset: str = "baseline",
        episode_source: str = "objectnav",
        source_kwargs: Optional[Dict[str, Any]] = None,
        slack_penalty: float = 0.0,
        collision_penalty: float = 0.0,
        seed: int = 0,
        logging_output_dir: Optional[str] = None,
        logger_actor: Any = None,
        **kwargs: Any,
    ):
        self.episodes_path = episodes
        self.scene_root = scene_root
        self.gap, self.dt = int(gap), float(dt)
        self.max_steps = int(max_steps)
        self.success_distance = float(success_distance)
        self.navmesh_choice, self.distance_to = navmesh, distance_to
        self.sensor_uuid, self.width, self.height = sensor_uuid, int(width), int(height)
        self.pid_preset = pid_preset
        self.episode_source_kind = episode_source
        self.source_kwargs = dict(source_kwargs or {})
        self.slack_penalty = float(slack_penalty)
        self.collision_penalty = float(collision_penalty)
        self.seed = int(seed)
        self.logging_output_dir = logging_output_dir
        self.logger_actor = logger_actor

        self._episodes: Optional[List[Any]] = None
        self._order: List[int] = []
        self._cursor = 0
        self._shard: Optional[List[str]] = None
        self._robot_sim = None
        self._scene_id: Optional[str] = None
        self._task = None
        self._executor = None
        self._screener = None
        self._episode = None
        self._steps = 0
        self._prev_geodesic = float("inf")
        self._skipped: Dict[str, int] = {}

    # -- env-actor interface ------------------------------------------------------------
    def assign_shard(self, episodes: Optional[List[str]] = None) -> None:
        """Take a slice of the split. `None` means everything this actor can see.

        ACCEPTS BOTH UID CONVENTIONS, because the two repos mean different things by
        `episode_id` and the difference is silent. `longnav.constants.episode_labels_table`
        labels look like `mv2HUxq3B53_55`, meaning *index 55 in that scene's JSON*, because
        habitat-lab's `from_json` overwrites `episode_id` with `str(i)`. `objectnav_eval`
        preserves the real id -- and in HM3D val every episode carries `episode_id == "0"` --
        so its uid for the same episode is `mv2HUxq3B53:0#56`. Matching only on our own uid
        would leave every longnav-labelled shard empty, and an empty shard reads as "this
        actor is already exhausted" rather than as an error.
        """
        self._shard = None if episodes is None else list(episodes)
        self._episodes = None
        self._cursor = 0

    def is_exhausted(self) -> bool:
        return self._episodes is not None and self._cursor >= len(self._order)

    def flush_logs_to_disk(self):
        return None

    def reset(self):
        self._load_episodes()
        episode, screen = self._next_admissible_episode()
        self._episode = episode
        self._ensure_scene(episode)
        start = self._task.reset(episode.episode, snap_to_navmesh=True)
        self._executor.reset()
        self._steps = 0
        self._prev_geodesic = float(start["distance_to_goal"])
        return self._render(), {
            "obs": {"instr_or_goal": episode.object_category},
            "reward": 0.0,
            "done": False,
            "is_exhausted": self.is_exhausted(),
            "info": {
                "episode_label": episode.uid,
                "distance_to_goal": self._finite(self._prev_geodesic),
                "screen": screen,
                "skipped_so_far": dict(self._skipped),
            },
        }

    def step(self, action, supplementary_logs: Optional[Dict[str, Any]] = None):
        """`action` is the `(gap, 3)` chunk prefix the policy head already truncated."""
        chunk = np.asarray(action, dtype=np.float64).reshape(-1, 3)
        if len(chunk) != self.gap:
            raise ValueError(
                f"env gap={self.gap} but received a {len(chunk)}-row chunk. The head and the "
                "env must agree on ticks-per-step, or sim time and policy steps quietly mean "
                "different things and two runs stop being comparable."
            )
        execution = self._executor.execute(chunk, chunk_index=self._steps)
        self._steps += 1

        # One geodesic query per policy step, not per tick: the task layer keeps path length
        # on the cheap PathTracker precisely so stepping never triggers a navmesh query.
        self._task.observe_pose(self._robot_sim.get_2d_pose())
        geodesic = float(self._task.evaluate()["distance_to_goal"])

        progress = (self._prev_geodesic - geodesic
                    if np.isfinite(geodesic) and np.isfinite(self._prev_geodesic) else 0.0)
        collided = self._collided(execution)
        reward = float(progress) - self.slack_penalty
        if collided:
            reward -= self.collision_penalty
        self._prev_geodesic = geodesic

        reached = bool(np.isfinite(geodesic) and geodesic <= self.success_distance)
        escaped = not np.isfinite(geodesic)
        done = bool(reached or escaped or self._steps >= self.max_steps)
        return self._render(), {
            "obs": {"instr_or_goal": self._episode.object_category},
            "reward": reward,
            "done": done,
            "is_exhausted": self.is_exhausted(),
            "info": {
                "distance_to_goal": self._finite(geodesic),
                "success": reached,
                "escaped": escaped,
                "steps": self._steps,
                "collided": collided,
                # How far the chunk asked the base to move this step. Paired with the
                # reward it separates "the policy commanded nothing" from "the policy
                # commanded a move the controller could not execute".
                "commanded_m": float(execution.ticks[-1].commanded_displacement)
                if execution.ticks else 0.0,
                "end_lag_m": float(execution.end_lag),
            },
        }

    # -- construction, done lazily inside the actor process ------------------------------
    # None of this can happen in __init__: Ray pickles the actor's arguments to the worker,
    # and a HabitatRobotSim is not picklable. It is also why the simulator is per-actor --
    # habitat-sim aborts the process at teardown when two Simulators are alive in one
    # process, which is fine here because Ray gives each actor its own.
    def _load_episodes(self) -> None:
        if self._episodes is not None:
            return
        from objectnav_eval.episodes import build_episode_source, env_scene_root

        self.scene_root = env_scene_root(self.scene_root)
        # `path` rather than `episodes`, and no `scene_root`: sources take the shard path
        # only, and scene resolution happens later in `resolve_scene`. Extra kwargs are
        # forwarded verbatim so a source needing more than a path does not add a field here.
        source = build_episode_source(
            self.episode_source_kind, path=self.episodes_path, **self.source_kwargs,
        )
        episodes = list(source.load())
        if self._shard is not None:
            episodes = self._select(episodes, self._shard)
        self._episodes = episodes
        rng = np.random.default_rng(self.seed)
        self._order = list(rng.permutation(len(episodes)))
        self._cursor = 0


    def _select(self, episodes: List[Any], wanted: List[str]) -> List[Any]:
        """Resolve a shard against harness uids, and RAISE on anything unresolved.

        DELIBERATELY DOES NOT TRANSLATE longnav labels. The two repos mean different things
        by `episode_id`: `longnav.constants.episode_labels_table` uses `mv2HUxq3B53_55`,
        meaning index 55 in that scene's JSON, while a harness uid is
        `scene:episode_id#occurrence` where the occurrence counts appearances of that
        `scene:episode_id` pair in load order. Those coincide only when every episode in the
        scene carries the same id -- true for most of HM3D val and NOT true everywhere, so
        deriving one from the other positionally resolves to the WRONG EPISODE in some
        scenes, silently, and the run reports a sample101 number over a set that is not
        sample101.

        `scripts/episode_set_from_longnav.py` in habitat_physical_nav already does this
        translation and VERIFIES each index against the raw JSON's `start_position` rather
        than assuming it. Run that once, pass its output here, and there is nothing to get
        wrong. Re-deriving it in this file would be a second implementation of exactly the
        mapping that needs checking.
        """
        by_uid = {e.uid: e for e in episodes}
        out, missing = [], []
        for label in wanted:
            hit = by_uid.get(label)
            if hit is None:
                missing.append(label)
            else:
                out.append(hit)
        if missing:
            looks_longnav = sum("_" in m and ":" not in m for m in missing)
            hint = ""
            if looks_longnav:
                hint = (
                    " -- these look like longnav labels (`scene_index`). Translate them "
                    "first with habitat_physical_nav/scripts/episode_set_from_longnav.py, "
                    "which checks each index against the raw JSON instead of assuming the "
                    "index and the uid occurrence agree; they do not in every scene."
                )
            raise KeyError(
                f"{len(missing)} of {len(wanted)} shard labels matched no episode in "
                f"{self.episodes_path} (e.g. {missing[:3]}){hint}"
            )
        return out

    def _next_admissible_episode(self):
        """The next episode the robot can actually reach, skipping the rest with a reason.

        A screened-out episode is NOT returned as a zero-reward rollout: the goal being on an
        unreachable island is a property of the navmesh, not of the policy, and training on it
        teaches that some goals cannot be reached at all.
        """
        from objectnav_eval.screening import EpisodeScreener

        while self._cursor < len(self._order):
            episode = self._episodes[self._order[self._cursor]]
            self._cursor += 1
            self._ensure_scene(episode)
            if self._screener is None:
                # Screened against the robot's own pathfinder -- the one every distance in
                # this env is measured on -- with the same success threshold the reward uses,
                # so an episode that starts already inside it is skipped rather than handing
                # the policy a success it did not navigate to.
                self._screener = EpisodeScreener(
                    robot_pathfinder=self._robot_sim.sim.pathfinder,
                    success_distance=self.success_distance,
                )
            screen = self._screener.screen(episode)
            if screen.ok:
                return episode, screen.code or "ok"
            code = screen.code or "unknown"
            self._skipped[code] = self._skipped.get(code, 0) + 1
        raise StopIteration(
            "this shard is exhausted; the rollout driver should call assign_shard again "
            f"(skipped by reason: {self._skipped})"
        )

    def _ensure_scene(self, episode) -> None:
        """One simulator per actor, rebuilt only when the scene changes."""
        from objectnav_eval.harness import build_scene_simulator

        if self._robot_sim is not None and episode.scene_id == self._scene_id:
            return
        if self._robot_sim is not None:
            # Switching in place rather than building a second Simulator: two alive in one
            # process aborts habitat-sim at teardown.
            self._robot_sim.reset(scene_id=episode.scene_id)
            self._scene_id = episode.scene_id
            self._rebuild_task_and_executor()
            return
        self._robot_sim, _ = build_scene_simulator(episode, self._sim_config())
        self._scene_id = episode.scene_id
        self._rebuild_task_and_executor()

    def _rebuild_task_and_executor(self) -> None:
        from continuous_demos.pid_pose_controller import PIDPoseController

        from objectnav_eval.control import build_pid_config
        from objectnav_eval.executor import ChunkExecutor
        from objectnav_eval.task import build_task

        self._task = build_task(
            self._robot_sim, success_distance=self.success_distance,
            distance_to=self.distance_to, scene_id=self._scene_id,
            scene_root=self.scene_root, live_height=True,
        )
        self._executor = ChunkExecutor(
            self._robot_sim, PIDPoseController(self._robot_sim,
                                               build_pid_config(self.pid_preset, None)),
            dt=self.dt, gap=self.gap, collect_contacts=self.collision_penalty > 0.0,
            log_camera=False, sensor_uuid=self.sensor_uuid,
        )
        self._screener = None

    def _sim_config(self):
        """The subset of the eval harness's config object `build_scene_simulator` reads."""
        from types import SimpleNamespace

        return SimpleNamespace(
            scene_root=self.scene_root, width=self.width, height=self.height,
            navmesh=self.navmesh_choice,
        )

    # -- small helpers -------------------------------------------------------------------
    def _render(self) -> np.ndarray:
        obs = self._robot_sim.get_obs()
        rgb = obs[self.sensor_uuid]
        return np.asarray(rgb, dtype=np.uint8)[..., :3]

    def _collided(self, execution) -> bool:
        if not self.collision_penalty:
            return False
        # `obstruction`, NOT the total. The robot is always in contact with the floor -- a
        # healthy tick sits at ~6 broadphase pairs and 2 contact points with nothing wrong --
        # so a raw count is true on every tick and the penalty would quietly degenerate into
        # a constant slack cost. The discriminator is the contact normal: standing on a floor
        # gives a near-vertical one, pressing into a wall a near-horizontal one, and
        # `ContactSnapshot` has already made that split.
        return any(getattr(t, "contacts", None) and t.contacts.obstruction > 0
                   for t in execution.ticks)

    @staticmethod
    def _finite(value: float):
        """Non-finite distances are legitimate (the robot left the world) but must serialise
        as `null`, not as `inf`, or the trajectory log stops being strict JSON."""
        return float(value) if np.isfinite(value) else None
