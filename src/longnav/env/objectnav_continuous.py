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
import os

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
        #self.seed = int(seed)
        # CRITICAL: need different seed per worker to sample properly, below is jank but more correct version:
        np.random.seed(os.getpid())
        self.seed = np.random.randint(0,100000)
        
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
        self._screen_pf = None
        self._screen_pf_scene: Optional[str] = None
        self._episode = None
        self._steps = 0
        self._prev_geodesic = float("inf")
        self._skipped: Dict[str, int] = {}
        # Per-episode cache backing `flush_logs_to_disk`. The rollout scheduler fires that
        # as a Ray remote after each episode (rollout_core.py:705), so video encoding runs
        # inside this actor while the scheduler dispatches elsewhere -- the write cost is
        # amortized across the fleet exactly as it is for the discrete actor.
        self.minimal_logging = bool(kwargs.get("minimal_logging", False))
        self._cache: Dict[str, list] = {"rgb": [], "info": [], "reward": []}

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

    def flush_logs_to_disk(self, clear_steps: bool = True):
        """Write this episode's video, summary and wandb payload; ship to the logger actor.

        Mirrors the discrete actor's contract -- `vid/episode_video`, `img/thumbnail` and
        scalar keys in one payload row, `results_{pid}` jsonl on disk -- so
        `logging_workers` wandb-wraps it identically and the oSR/oSPL growth curves land
        beside the discrete runs' SR/SPL. The overlay text is continuous-appropriate
        (distance, reward, commanded displacement) rather than the discrete action table.
        """
        import json as _json
        import os
        import time as _time

        if self.logging_output_dir is None or not self._cache["info"]:
            if clear_steps:
                self._cache = {"rgb": [], "info": [], "reward": []}
            return None
        infos, rewards = self._cache["info"], self._cache["reward"]
        first, last = infos[0], infos[-1]
        ep_label = str(first.get("episode_label", f"ep_{_time.time():.0f}"))
        scene = str(first.get("scene_id", "unknown_scene")).replace("/", "_")
        save_dir = os.path.join(self.logging_output_dir, scene,
                                f"{ep_label}.{os.getpid()}@{_time.time():.0f}")
        os.makedirs(save_dir, exist_ok=True)

        n = max(len(rewards), 1)
        episode_logs: Dict[str, Any] = {
            "episode_label": ep_label, "scene_id": first.get("scene_id"),
            "goal": self._episode.object_category if self._episode else None,
            "n_steps": len(rewards),
            "success": bool(last.get("success", False)),
            "oracle_success": bool(last.get("oracle_success", False)),
            "ospl_fix": float(last.get("ospl_fix", 0.0)),
            "min_m": last.get("min_m"), "start_m": last.get("start_m"),
            "path_length_m": last.get("path_length_m"),
            "escaped": bool(last.get("escaped", False)),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "collision_rate": float(np.mean([bool(i.get("collided")) for i in infos])),
            "worker_pid": os.getpid(), "timestamp": _time.time(),
        }
        if not self.minimal_logging and self._cache["rgb"]:
            try:
                episode_logs |= self._write_video(save_dir)
            except Exception as e:      # video failure must never kill a training episode
                print(f"[objectnav_continuous] video write failed: {e}")
        seq = {
            "distance_to_goal": [i.get("distance_to_goal") for i in infos],
            "positions": [i.get("pos_rots", [None] * 3)[:3] for i in infos],
            "reward": rewards,
        }
        def _ser(o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)
        with open(os.path.join(save_dir, "sequence.json"), "w") as f:
            _json.dump(seq, f, default=_ser)
        with open(os.path.join(save_dir, "summary.json"), "w") as f:
            _json.dump(episode_logs, f, default=_ser, indent=2)
        with open(os.path.join(self.logging_output_dir, f"results_{os.getpid()}"), "a") as f:
            f.write(_json.dumps(episode_logs, default=_ser) + "\n")
        if clear_steps:
            self._cache = {"rgb": [], "info": [], "reward": []}
        if self.logger_actor is not None:
            import ray
            try:
                ray.get(self.logger_actor.log_row.remote(row=episode_logs), timeout=1.0)
            except Exception as e:
                print(f"[objectnav_continuous] logger ack issue: {e}")
        return os.path.join(save_dir, "summary.json")

    def _write_video(self, save_dir: str) -> Dict[str, str]:
        """One MP4 per episode with a continuous-status overlay, plus a thumbnail."""
        import os
        from habitat.utils.visualizations import utils as vut
        from PIL import Image

        infos = self._cache["info"]
        frames = []
        for idx, (rgb, info) in enumerate(zip(self._cache["rgb"], infos)):
            r = self._cache["reward"][idx] if idx < len(self._cache["reward"]) else 0.0
            text = [
                f"episode: {infos[0].get('episode_label')} step: {idx}",
                f"goal: {self._episode.object_category if self._episode else '?'}",
                f"distance_to_goal: {info.get('distance_to_goal')}",
                f"reward: {r:+.3f}  commanded_m: {info.get('commanded_m', 0.0):.2f}",
            ]
            frames.append(vut.overlay_text_to_image(np.ascontiguousarray(rgb), text))
        vut.images_to_video(images=frames, output_dir=save_dir, video_name="video",
                            fps=4, quality=4, verbose=False)
        thumb = os.path.join(save_dir, "thumbnail.jpg")
        Image.fromarray(frames[-1]).save(thumb, quality=85)
        return {"vid/episode_video": os.path.join(save_dir, "video.mp4"),
                "img/thumbnail": thumb}

    def reset(self):
        self._load_episodes()
        episode, screen = self._next_admissible_episode()
        self._episode = episode
        self._ensure_scene(episode)
        start = self._task.reset(episode.episode, snap_to_navmesh=True)
        self._executor.reset()
        self._steps = 0
        self._prev_geodesic = float(start["distance_to_goal"])
        # Oracle bookkeeping, so every episode emits the STOP-INDEPENDENT metrics the
        # project actually reads (docs/SAMPLE101_EVALS.md): with no stop head, `success`
        # measures the termination rule, while oracle_success / ospl_fix measure
        # navigation. These are what make wandb growth curves comparable -- loosely --
        # against the discrete runs' SR/SPL.
        self._start_geodesic = self._prev_geodesic
        self._min_geodesic = self._prev_geodesic
        self._path_at_min = 0.0
        rgb = self._render()
        info = {
            "episode_label": episode.uid,
            "scene_id": getattr(episode, "scene_id", None) or self._scene_id,
            "distance_to_goal": self._finite(self._prev_geodesic),
            "pos_rots": self._pos_rots(),
            "screen": screen,
            "skipped_so_far": dict(self._skipped),
        }
        self._cache = {"rgb": [rgb], "info": [info], "reward": []}
        return rgb, {
            "obs": {"instr_or_goal": episode.object_category},
            "reward": 0.0,
            "done": False,
            "is_exhausted": self.is_exhausted(),
            "info": info,
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

        if np.isfinite(geodesic) and geodesic < self._min_geodesic:
            self._min_geodesic = float(geodesic)
            self._path_at_min = float(self._task.path_tracker.length)

        reached = bool(np.isfinite(geodesic) and geodesic <= self.success_distance)
        escaped = not np.isfinite(geodesic)
        done = bool(reached or escaped or self._steps >= self.max_steps)
        info_extra: Dict[str, Any] = {}
        if done:
            # Emitted once, on the terminal step, so a per-episode reduction upstream
            # (last-info wins) reads them directly. `ospl_fix` is the RECOMPUTED oracle
            # SPL -- the task layer's own OracleSPL is documented broken
            # (docs/SAMPLE101_EVALS.md) and is deliberately not consulted here.
            oracle = bool(self._min_geodesic <= self.success_distance)
            s0 = self._start_geodesic
            info_extra = {
                "oracle_success": oracle,
                "ospl_fix": (s0 / max(s0, self._path_at_min)) if oracle and s0 > 0 else 0.0,
                "min_m": self._finite(self._min_geodesic),
                "start_m": self._finite(s0),
                "path_length_m": float(self._task.path_tracker.length),
            }
        rgb = self._render()
        info = {
            "episode_label": self._episode.uid,
            "scene_id": getattr(self._episode, "scene_id", None) or self._scene_id,
            "distance_to_goal": self._finite(geodesic),
            "success": reached,
            "escaped": escaped,
            "steps": self._steps,
            "collided": collided,
            "stuck": collided,        # discrete-actor key parity: collision_rate reads it
            "pos_rots": self._pos_rots(),
            **info_extra,
            # How far the chunk asked the base to move this step. Paired with the
            # reward it separates "the policy commanded nothing" from "the policy
            # commanded a move the controller could not execute".
            "commanded_m": float(execution.ticks[-1].commanded_displacement)
            if execution.ticks else 0.0,
            "end_lag_m": float(execution.end_lag),
        }
        self._cache["rgb"].append(rgb)
        self._cache["info"].append(info)
        self._cache["reward"].append(float(reward))
        return rgb, {
            "obs": {"instr_or_goal": self._episode.object_category},
            "reward": reward,
            "done": done,
            "is_exhausted": self.is_exhausted(),
            "info": info,
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
        kwargs = dict(self.source_kwargs)
        if (self._shard and "scenes" not in kwargs
                and self.episode_source_kind == "objectnav"):
            # The shard's uids name their scenes (`scene:episode_id#occurrence`), and the
            # full HM3D train split is 133 MB gzipped / minutes of parse. Restrict the load
            # to the shard's own scenes -- measured: the eager full-split load exceeded
            # four minutes PER ACTOR and was the entire "hung at Starting rollout
            # collection" incident.
            kwargs["scenes"] = sorted({u.split(":", 1)[0] for u in self._shard if ":" in u})
        source = build_episode_source(
            self.episode_source_kind, path=self.episodes_path, **kwargs,
        )
        episodes = list(source.load())
        if self._shard is not None:
            episodes = self._select(episodes, self._shard)
        self._episodes = episodes
        rng = np.random.default_rng(self.seed)
        # SCENE-GROUPED order, never a flat permutation. A flat shuffle makes consecutive
        # episodes land in different scenes, and every scene change costs a full simulator
        # reconfigure PLUS a robot-footprint navmesh recompute -- ~30 s of pure overhead per
        # episode, forever. Shuffle scenes, then episodes within a scene: same seed-determined
        # coverage, one scene build per scene visit. (CLAUDE.md's task layer gives the same
        # advice: drive the outer loop with episodes_by_scene.)
        by_scene: Dict[str, List[int]] = {}
        for i, e in enumerate(episodes):
            by_scene.setdefault(e.uid.split(":", 1)[0], []).append(i)
        scene_order = list(by_scene)
        rng.shuffle(scene_order)
        order: List[int] = []
        for s in scene_order:
            idx = by_scene[s]
            rng.shuffle(idx)
            order.extend(idx)
        self._order = order
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
            pf = self._screening_pathfinder()
            if self._screener is None or self._screener.robot_pathfinder is not pf:
                # Screened against the ROBOT's reachability even when rewards are measured
                # on the dataset mesh. Under `navmesh="dataset"` the sim's pathfinder holds
                # the shipped point-agent mesh -- connected, unfragmented, and an OVERSTATED
                # answer to "where can a 0.28 m robot go" (16/30 val_mini goals sit on a
                # different island of the robot mesh). Screening against it would admit
                # episodes the robot cannot complete, which enter training as zero-progress
                # rollouts and teach the policy that some goals simply pay nothing. So the
                # screen runs on a second, robot-footprint pathfinder; the reward keeps the
                # dataset mesh, whose geodesic is smooth -- the robot mesh's island snapping
                # injects multi-metre distance jumps (docs/SAMPLE101_EVALS.md), which a
                # per-step progress REWARD amplifies far more than an eval metric does.
                self._screener = EpisodeScreener(
                    robot_pathfinder=pf,
                    dataset_pathfinder=(self._robot_sim.sim.pathfinder
                                        if self.navmesh_choice == "dataset" else None),
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
    def _screening_pathfinder(self):
        """The pathfinder that answers "can the robot reach this goal", per scene.

        With `navmesh="robot"` the sim's own pathfinder already is the robot mesh. With
        `navmesh="dataset"` a second pathfinder is recomputed for the robot footprint and
        cached per scene -- rebuilt on scene switch, because a screener holding the previous
        scene's mesh would screen every episode against the wrong building."""
        if self.navmesh_choice != "dataset":
            return self._robot_sim.sim.pathfinder
        if self._screen_pf_scene != self._scene_id:
            import habitat_sim
            from objectnav_eval.navmesh import (ROBOT_NAVMESH_HEIGHT,
                                                ROBOT_NAVMESH_RADIUS)
            pf = habitat_sim.nav.PathFinder()
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            settings.agent_radius = ROBOT_NAVMESH_RADIUS
            settings.agent_height = ROBOT_NAVMESH_HEIGHT
            if not self._robot_sim.sim.recompute_navmesh(pf, settings) or not pf.is_loaded:
                raise RuntimeError(
                    f"robot-footprint navmesh recompute failed for {self._scene_id}; "
                    "refusing to screen on the point-agent mesh instead, which would "
                    "silently admit unreachable episodes into training."
                )
            self._screen_pf, self._screen_pf_scene = pf, self._scene_id
        return self._screen_pf

    def _pos_rots(self) -> List[float]:
        """Discrete-actor `pos_rots` parity: 7 floats, position then quaternion.

        PLANAR CONTROL FRAME, not habitat world -- `[X, Y, 0]` plus a yaw-about-+z
        quaternion from theta. The consumer plots `[:3]` as the trajectory and stores the
        rest; nothing downstream inverts frames, so the planar one is the honest choice."""
        x, y, th = self._robot_sim.get_2d_pose()
        return [float(x), float(y), 0.0,
                float(np.cos(th / 2)), 0.0, 0.0, float(np.sin(th / 2))]

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
