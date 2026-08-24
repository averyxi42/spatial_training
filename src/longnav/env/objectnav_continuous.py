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

from typing import Any, Dict, List, Optional, Sequence

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
        progress_reward_clip: float = 0.75,
        success_reward: float = 0.0,
        escape_penalty: float = 0.0,
        reward_lost_steps: int = 25,
        exclude_categories: Optional[Sequence[str]] = None,
        train_uids: Optional[Any] = None,
        seed: Optional[int] = None,
        logging_output_dir: Optional[str] = None,
        logger_actor: Any = None,
        video_tick_stride: int = 2,
        video_realtime_factor: float = 1.0,
        rtc_delay_source: str = "none",
        rtc_delay: int = 0,
        rtc_delay_max: int = 0,
        rtc_delay_base: float = 0.8,
        rtc_delay_seed: int = 0,
        rtc_overrun_rate: float = 0.0,
        snap_start: bool = True,
        screen_reachability: bool = True,
        tick_metrics: bool = False,
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
        # Physical-consistency bound on the progress REWARD (metrics are untouched). The
        # geodesic is queried from a navmesh snap of a physically simulated robot, and the
        # snap occasionally relocates across a wall or floor: measured over 506 episodes
        # of flow_sde_train_fixed_lr, 0.16% of steps carried a persistent >1 m single-step
        # geodesic jump (max 6.7 m -- ~20x a normal step reward), touching 10% of
        # episodes. No physical step can change the geodesic by more than the path length
        # driven in gap*dt seconds (~<=0.6 m), so anything beyond the clip is a
        # measurement artifact, not motion. 0 disables.
        self.progress_reward_clip = float(progress_reward_clip)
        # Terminal bonus on the reached step (the discrete standard's success term). At a
        # tight success_distance the progress reward alone barely distinguishes "arrived"
        # from "hovered nearby" -- the last 0.2 m pays ~0.2 -- so precision termination
        # needs its own signal. 0 (default) is the pre-existing progress-only shape.
        self.success_reward = float(success_reward)
        # Penalty on the escaped terminal step. "Escaped" is the PHYSICAL fall detector
        # from SFT data collection (continuous_demos.drive_failure: joint_z free-fall
        # sustained fall_duration_s), never the geodesic -- under the earlier
        # `not isfinite(geodesic)` proxy, 42% of the 250st run's "escapes" were spawn-snap
        # failures with zero policy involvement. Unpenalized, a real escape ends the
        # episode keeping all accumulated progress reward -- a free exit. Positive value =
        # subtracted from the escaped step's reward. 0 keeps the old shape.
        self.escape_penalty = float(escape_penalty)
        # Sustained-blindness gate: reset-time DOA screening checks connectivity at the
        # SPAWN SNAP, but an episode can go permanently unmeasurable the moment the robot
        # moves (measured: 1S7LAXRdDqK:48 -- start_m finite at 6.89 m, then geodesic
        # non-finite for the whole episode; the finite-hold froze reward at ~0 over a
        # 44 m wander, 0/29 forever). N consecutive policy steps of non-finite geodesic
        # ends the episode early with `reward_lost` flagged: transient snap dropouts span
        # ticks, not tens of steps, so 25 steps (10 s of sim) is far above their scale.
        # 0 disables.
        self.reward_lost_steps = int(reward_lost_steps)
        # Goal categories to drop from this actor's pool entirely. EMPTY BY DEFAULT, and
        # the empty case is not merely a no-op filter -- `_load_episodes` skips the pass
        # outright, so a run without it loads exactly the episodes it always did.
        # Motivating case: `plant` is often mislabelled in HM3D, so a policy is punished
        # for not finding a thing that is not there.
        # Restrict the TRAINING stream to these uids (a list, or a path to a
        # comma/newline-separated file). The pool itself is untouched, which is the
        # point: uids are occurrence-counted per shard file, so building a smaller
        # DATASET would renumber them and silently break any pinned eval set. Applies
        # only when no explicit shard is assigned -- `run_eval_cycle` assigns eval uids
        # and restores with `assign_shard(None)`, so eval still reaches episodes that
        # training never serves. That is what makes a disjoint held-out eval possible.
        self._train_uids = None
        if train_uids:
            if isinstance(train_uids, str):
                raw = open(train_uids).read()
                train_uids = [u.strip() for u in raw.replace("\n", ",").split(",")]
            self._train_uids = frozenset(u for u in train_uids if u)
        self._exclude_categories = frozenset(
            c.strip().lower() for c in (exclude_categories or ()) if c and c.strip()
        )
        # Spawn-snap policy. True is this env's historical behaviour; the EVAL HARNESS
        # does NOT snap (its snap_start defaults False), and a snapped spawn can land
        # disconnected from the goal on the dataset mesh -- measured here as ~30% of the
        # pinned sample101 set skipped as reward_mesh_disconnected, episodes the harness
        # serves fine. Cross-path parity evals must set False; training runs keep True
        # unless deliberately re-based (changing it changes start distances and every
        # downstream number).
        self.snap_start = bool(snap_start)
        # Robot-footprint reachability screening. True (historical) for training --
        # unreachable goals enter training as zero-progress rollouts and teach that some
        # goals pay nothing. False = the harness's serve-everything convention, required
        # for cross-path parity on pinned sets (see _next_admissible_episode).
        self.screen_reachability = bool(screen_reachability)
        # Per-tick metric latching (the harness's evaluate_every=1). False keeps the
        # one-geodesic-query-per-step training economy; parity evals set True -- see
        # _metrics_on_tick.
        self.tick_metrics = bool(tick_metrics)
        # -- RTC latency masking (docs/RTC_RL.md; schedule from objectnav_eval) --------
        # "none" (default) is the historical env, bit for bit: no scheduler is ever
        # constructed, step() keeps its (gap, 3) contract, and no extra geodesic query
        # runs. Any other source turns on the reciprocal schedule: the head sends the
        # FULL (n_ticks, 3) chunk, this actor owns the slicing, the obs carries the
        # committed prefix, and each step's reward is split at the splice point
        # (r_commit / r_fresh in info) so returns can be re-timed on the trainer side.
        self._rtc_source = str(rtc_delay_source)
        self._rtc = self._rtc_source != "none"
        self._rtc_delay = int(rtc_delay)
        self._rtc_delay_max = int(rtc_delay_max)
        self._rtc_delay_base = float(rtc_delay_base)
        self._rtc_delay_seed = int(rtc_delay_seed)
        self._rtc_overrun_rate = float(rtc_overrun_rate)
        self._scheduler = None
        self._rtc_episode_index = 0
        self._commit_done = False
        self._geo_after_commit: Optional[float] = None
        self._interval_execs: list = []
        # The SFT rejection thresholds, verbatim; imported here so a stale fork of the
        # numbers cannot drift from the corpus generator's.
        from continuous_demos.drive_failure import DriveFailureConfig
        self._fall_cfg = DriveFailureConfig()
        self._fall_run = 0
        self._max_fall_run = 0
        # Per-worker seeding, the discrete env's `ep_seed` pattern (habitat.py:629-634):
        # an explicit seed reproduces an exact episode stream (eval); None -- the RL
        # default -- derives a distinct seed per worker process, so parallel actors do not
        # all replay the same episode order.
        if seed is None:
            np.random.seed(os.getpid())
            self.seed = int(np.random.randint(0, 100000))
        else:
            self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        # Video capture cadence. Frames are taken every `video_tick_stride` PHYSICS ticks
        # (1/dt = 25 Hz base) through the executor's on_tick hook, JPEG-encoded in memory
        # as they arrive (~40 KB/frame; raw 640x480 frames at stride 1 would be ~1.6 GB an
        # episode). Written fps = realtime_factor / (dt * stride): 1.0 is realtime at any
        # stride, bigger is faster playback.
        self._video_tick_stride = int(video_tick_stride)
        if self._video_tick_stride < 1:
            raise ValueError(f"video_tick_stride must be >= 1, got {video_tick_stride}")
        self._video_realtime_factor = float(video_realtime_factor)
        if not self._video_realtime_factor > 0:
            raise ValueError(
                f"video_realtime_factor must be > 0, got {video_realtime_factor}")
        self._video_frames: List[bytes] = []
        self._video_meta: List[tuple] = []
        self._video_tick = 0

        self.logging_output_dir = logging_output_dir
        self.logger_actor = logger_actor
        # Metric routing: interleaved eval sets this to "eval_env/" for its episodes so
        # ODE-eval rows never interleave with SDE-training rows on the same charts.
        self._log_prefix = ""

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
        new = None if episodes is None else list(episodes)
        if self._episodes is not None and new == self._shard:
            # The same shard handed back (the RL driver cycles one pool): keep the parsed
            # episodes -- a full-split re-parse is minutes per actor -- and just deal a
            # fresh permutation for the next pass.
            self._reshuffle()
            return
        self._shard = new
        self._episodes = None
        self._cursor = 0

    def is_exhausted(self) -> bool:
        """True when this actor NEEDS WORK: no episodes loaded yet, or the assigned
        shard is spent. THE FRESH STATE IS EXHAUSTED -- this is the discrete env's
        contract (`habitat.py`: `episode_counter >= shard_length` with both 0 at
        init), and it is what `collect_rollouts`' bootstrap keys on to hand each
        actor its initial shard.

        History (2026-08-24): this method previously returned False for a fresh
        actor (`_episodes is not None and ...`), which silently defeated the shard
        bootstrap for the continuous env in EVERY flow -- training never consumed
        its shard iterator (each actor free-streamed its own pool; `train_uids` was
        layered on as a de facto workaround), and standalone eval never served its
        pinned subsets (observed: 128 collections, 29 unique episodes, none
        guaranteed from the requested set). Restoring the contract is safe for
        training only together with the trivial-shard default
        (`get_shard_iterator`'s no-source fallback + `subset_label` defaulting to
        empty): a trivial shard (None) means "load the full split yourself", which
        reproduces the previous de facto training stream through the front door.
        """
        return self._episodes is None or self._cursor >= len(self._order)

    def set_log_prefix(self, prefix: str) -> None:
        """Prefix for wandb metric keys of subsequent episodes ("" = training stream)."""
        self._log_prefix = str(prefix or "")

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
                self._video_frames, self._video_meta = [], []
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
            # int, not bool: wandb renders booleans in the run table only -- a bool metric
            # never gets a chart, and oracle_success IS the headline growth curve.
            "success": int(bool(last.get("success", False))),
            "oracle_success": int(bool(last.get("oracle_success", False))),
            "ospl_fix": float(last.get("ospl_fix", 0.0)),
            "min_m": last.get("min_m"), "start_m": last.get("start_m"),
            "path_length_m": last.get("path_length_m"),
            "escaped": int(bool(last.get("escaped", False))),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "collision_rate": float(np.mean([bool(i.get("collided")) for i in infos])),
            "worker_pid": os.getpid(), "timestamp": _time.time(),
        }
        if not self.minimal_logging and self._video_frames:
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
        def _denan(o):
            """NaN -> null at SERIALISATION time only: step info keeps NaN (a float,
            collatable); JSON keeps strict null. See `_finite`."""
            if isinstance(o, float) and not np.isfinite(o): return None
            if isinstance(o, dict): return {k: _denan(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)): return [_denan(v) for v in o]
            return o
        with open(os.path.join(save_dir, "sequence.json"), "w") as f:
            _json.dump(_denan(seq), f, default=_ser)
        with open(os.path.join(save_dir, "summary.json"), "w") as f:
            _json.dump(_denan(episode_logs), f, default=_ser, indent=2)
        with open(os.path.join(self.logging_output_dir, f"results_{os.getpid()}"), "a") as f:
            f.write(_json.dumps(episode_logs, default=_ser) + "\n")
        if clear_steps:
            self._cache = {"rgb": [], "info": [], "reward": []}
            self._video_frames, self._video_meta = [], []
        if self.logger_actor is not None:
            import ray
            try:
                # Media namespaces (vid/, img/) must stay as-is for the logger's
                # detection; everything else takes the routing prefix.
                _row = {k if k.startswith(("vid/", "img/")) else self._log_prefix + k: v
                        for k, v in episode_logs.items()} if self._log_prefix else episode_logs
                ray.get(self.logger_actor.log_row.remote(row=_row), timeout=1.0)
            except Exception as e:
                print(f"[objectnav_continuous] logger ack issue: {e}")
        return os.path.join(save_dir, "summary.json")

    @property
    def _video_capture(self) -> bool:
        return not self.minimal_logging and self.logging_output_dir is not None

    def _encode_frame(self, rgb: np.ndarray) -> bytes:
        from io import BytesIO
        from PIL import Image

        buf = BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(
            buf, format="JPEG", quality=88)
        return buf.getvalue()

    def _metrics_on_tick(self, record) -> None:
        """Per-TICK metric latching, the harness's convention (evaluate_every=1).

        Off by default: a geodesic query per tick is exactly the cost the per-step
        design avoids in training. Cross-path parity evals need it -- committed motion
        swings more, and within-interval dips through the goal radius are latched by
        the harness's tick-granular oracle and missed by a step-granular one (the miss
        rate grows with d). Also feeds the path tracker per tick, which is what makes
        path-length (and therefore oSPL) comparable across stacks."""
        self._video_on_tick(record)
        self._task.observe_pose(record.pose)
        geo = float(self._task.evaluate()["distance_to_goal"])
        if np.isfinite(geo) and geo < self._min_geodesic:
            self._min_geodesic = geo
            self._path_at_min = float(self._task.path_tracker.length)

    @property
    def _on_tick(self):
        return self._metrics_on_tick if self.tick_metrics else self._video_on_tick

    def _video_on_tick(self, record) -> None:
        """Executor tick hook: capture every `video_tick_stride`-th physics tick.

        The counter is per-episode and global across chunks, so the stride's phase never
        resets at a chunk boundary and frame spacing stays exactly uniform. Overlay data is
        resolved at WRITE time (metrics are per policy step by design -- a geodesic query
        per tick is precisely what the task layer avoids), so the meta stores only which
        policy step the tick belongs to and its sim time."""
        # Physical fall tracking, EVERY tick regardless of video: the SFT rejection
        # mechanism (continuous_demos.drive_failure), verbatim thresholds -- falling is
        # joint_z velocity <= -fall_speed_mps (2.0, free fall past habitat's 0.2 m
        # agent_max_climb), escaped is that sustained fall_duration_s (1.0 s; measured
        # gap: longest healthy run 0.56 s, shortest true escape 1.6 s).
        from continuous_demos.drive_failure import vertical_velocity_mps
        if vertical_velocity_mps(self._robot_sim) <= -self._fall_cfg.fall_speed_mps:
            self._fall_run += 1
            self._max_fall_run = max(self._max_fall_run, self._fall_run)
        else:
            self._fall_run = 0

        self._video_tick += 1
        if self._video_tick % self._video_tick_stride:
            return
        if self._video_capture:
            self._video_frames.append(self._encode_frame(self._render()))
            self._video_meta.append((self._steps, float(record.sim_time)))

    def _write_video(self, save_dir: str) -> Dict[str, str]:
        """One MP4 per episode with a continuous-status overlay, plus a thumbnail.

        Written fps = realtime_factor / (dt * stride): at the 25 Hz physics rate, stride 1
        and factor 1.0 give a 25 fps video that plays in exact realtime; stride 5 gives a
        5 fps video that STILL plays realtime, just at lower temporal resolution."""
        import os
        from io import BytesIO
        from habitat.utils.visualizations import utils as vut
        from PIL import Image

        infos, rewards = self._cache["info"], self._cache["reward"]
        frames = []
        for jpeg, (step_idx, sim_time) in zip(self._video_frames, self._video_meta):
            # np.array (copy), NOT np.asarray: PIL exposes a READ-ONLY buffer, and cv2's
            # overlay refuses to draw into it ("marked as readonly"); ascontiguousarray
            # passes an already-contiguous readonly array through unchanged.
            rgb = np.array(Image.open(BytesIO(jpeg)).convert("RGB"), dtype=np.uint8)
            # A tick inside step k lands between infos[k] (pre-chunk) and infos[k+1]
            # (post-chunk); overlay the freshest bound that exists.
            k = min(step_idx + 1, len(infos) - 1)
            info = infos[k]
            r = rewards[k - 1] if 0 < k <= len(rewards) else 0.0
            text = [
                f"episode: {infos[0].get('episode_label')} step: {max(step_idx, 0)} "
                f"t={sim_time:.2f}s",
                f"goal: {self._episode.object_category if self._episode else '?'}",
                f"distance_to_goal: {info.get('distance_to_goal')}",
                f"reward: {r:+.3f}  commanded_m: {info.get('commanded_m', 0.0):.2f}",
            ]
            frames.append(vut.overlay_text_to_image(np.ascontiguousarray(rgb), text))
        fps = self._video_realtime_factor / (self.dt * self._video_tick_stride)
        vut.images_to_video(images=frames, output_dir=save_dir, video_name="video",
                            fps=fps, quality=4, verbose=False)
        thumb = os.path.join(save_dir, "thumbnail.jpg")
        Image.fromarray(frames[-1]).save(thumb, quality=85)
        return {"vid/episode_video": os.path.join(save_dir, "video.mp4"),
                "img/thumbnail": thumb}

    def list_episode_uids(self):
        """All uids this actor can currently serve (loads the pool if needed). Used by
        interleaved eval to draw the fixed eval set without a second parse."""
        self._load_episodes()
        return [e.uid for e in self._episodes]

    def reset(self):
        self._load_episodes()
        while True:
            try:
                episode, screen = self._next_admissible_episode()
            except StopIteration:
                # Shard exhausted DURING a skip loop (DOA tail at slice end): a raise
                # from a ray remote would error the reset ref and cascade into the
                # collection loop as a crash. Return a SENTINEL the rollout mixin
                # recognizes and retires on instead -- exhaustion is a normal outcome
                # of a finite eval slice, not an error.
                rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                return rgb, {
                    "obs": {"instr_or_goal": None},
                    "reward": 0.0, "done": True, "is_exhausted": True,
                    "exhausted_sentinel": True,
                    "info": {"exhausted_sentinel": True,
                             "skipped_so_far": dict(self._skipped)},
                }
            self._episode = episode
            self._ensure_scene(episode)
            start = self._task.reset(episode.episode, snap_to_navmesh=self.snap_start)
            if np.isfinite(float(start["distance_to_goal"])):
                break
            # Screening passes on the ROBOT-footprint mesh, but the REWARD is measured on
            # the dataset mesh -- and the spawn's dataset-mesh snap can land disconnected
            # from the goal (measured: 31/73 "escapes" in the 250st run were exactly this,
            # dead on arrival at n_steps=1). An unmeasurable episode is a screening
            # failure, not a rollout: skip it with a reason code, never serve it.
            self._skipped["reward_mesh_disconnected"] = (
                self._skipped.get("reward_mesh_disconnected", 0) + 1)
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
        self._cache = {"rgb": [], "info": [info], "reward": []}
        self._fall_run = 0
        self._max_fall_run = 0
        self._blind_run = 0
        self._video_frames, self._video_meta, self._video_tick = [], [], 0
        if self._video_capture:
            self._video_frames.append(self._encode_frame(rgb))
            self._video_meta.append((-1, 0.0))   # pre-motion frame; overlays info[0]
        obs: Dict[str, Any] = {"instr_or_goal": episode.object_category}
        if self._rtc:
            # Fresh scheduler per episode, stream seeded seed + episode index (the
            # delay-source discipline, LATENCY_MASKING.md section 4). The first
            # observe() forces d = 0 -- there is no tail at spawn -- so the first
            # decision is unconditioned by construction.
            self._scheduler = self._make_scheduler()
            self._scheduler.reset(self._rtc_episode_index)
            self._rtc_episode_index += 1
            self._commit_done = False
            self._geo_after_commit = None
            self._interval_execs = []
            pfx, d0 = self._scheduler.observe(self._robot_sim.get_2d_pose())
            obs["rtc_prefix"] = pfx
            obs["rtc_delay"] = int(d0)
        return rgb, {
            "obs": obs,
            "reward": 0.0,
            "done": False,
            "is_exhausted": self.is_exhausted(),
            "info": info,
        }

    # -- RTC latency masking (docs/RTC_RL.md; the schedule is objectnav_eval's) -------
    def _make_scheduler(self):
        from objectnav_eval.schedule import ChunkScheduler, DelaySource
        return ChunkScheduler(
            self.gap,
            DelaySource(self._rtc_source, delay=self._rtc_delay,
                        delay_max=self._rtc_delay_max, base=self._rtc_delay_base,
                        seed=self._rtc_delay_seed),
            overrun_rate=self._rtc_overrun_rate, seed=self._rtc_delay_seed,
        )

    def _execute_commitment(self) -> None:
        """Run the pending committed ticks and take the SPLICE-POINT geodesic reading
        (the reward split's mid measurement, docs/RTC_RL.md section 4). Runs at most
        once per decision; reached from begin_interval (overlap) or step (sync) --
        whichever comes first -- so the trajectory is identical either way."""
        committed = self._scheduler.pending_commitment()
        if committed is None:
            raise RuntimeError("no pending commitment: the scheduler has not observed")
        if len(committed):
            self._interval_execs.append(self._executor.execute_setpoints(
                committed, chunk_index=self._steps, on_tick=self._on_tick))
            self._task.observe_pose(self._robot_sim.get_2d_pose())
            self._geo_after_commit = float(self._task.evaluate()["distance_to_goal"])
        else:
            self._geo_after_commit = None      # d = 0: the splice IS the interval start
        self._commit_done = True

    def begin_interval(self) -> None:
        """Execute the committed ticks ahead of the chunk's arrival -- the opportunistic
        overlap entry (docs/RTC_RL.md section 5). Fired without awaiting by the rollout
        loop; Ray actor task ordering serializes the coming step() behind this call, so
        it changes wall-clock and nothing else. Raises when rtc is off rather than
        no-opping: a silently inert overlap flag is indistinguishable from a working one.
        """
        if not self._rtc or self._scheduler is None:
            raise RuntimeError(
                "begin_interval requires rtc_delay_source != 'none' on this env")
        if not self._commit_done:
            self._execute_commitment()

    def step(self, action, supplementary_logs: Optional[Dict[str, Any]] = None):
        """`action`: without RTC, the `(gap, 3)` chunk prefix the policy head already
        truncated -- the historical contract, unchanged. With RTC, the FULL
        `(n_ticks, 3)` chunk: this actor owns the slicing (docs/RTC_RL.md section 5) --
        the scheduler tiles the interval as committed ticks + fresh span and retains
        the tail as the next commitment's source."""
        chunk = np.asarray(action, dtype=np.float64).reshape(-1, 3)
        base_geo = self._prev_geodesic
        if not self._rtc:
            if len(chunk) != self.gap:
                raise ValueError(
                    f"env gap={self.gap} but received a {len(chunk)}-row chunk. The head and the "
                    "env must agree on ticks-per-step, or sim time and policy steps quietly mean "
                    "different things and two runs stop being comparable."
                )
            execution = self._executor.execute(
                chunk, chunk_index=self._steps,
                # The on_tick hook carries the per-tick concerns: physical fall tracking
                # (always), video frames (when capture is on), and -- under tick_metrics
                # -- the harness-convention per-tick metric latch.
                on_tick=self._on_tick,
            )
            executions = [execution]
            geo_mid = None
        else:
            if len(chunk) <= self.gap:
                raise ValueError(
                    f"rtc needs the FULL chunk (> gap={self.gap} rows) so the tail can "
                    f"fund the next commitment; got {len(chunk)} rows. The head returns "
                    "the full block when called with a prefix -- rollout_core passes "
                    "obs['rtc_prefix']."
                )
            if not self._commit_done:
                self._execute_commitment()
            plan = self._scheduler.accept(chunk)
            fresh = plan.setpoints[plan.committed:]
            if len(fresh):
                # The fresh span CHAINS its feedforward from the last COMMITTED
                # setpoint -- the commanded chain, exactly what the harness's single
                # merged block does. Re-seeding from the live pose here (the default)
                # would inject a tracking-lag transient into the feedforward at every
                # splice, a control divergence the merged-block path does not have.
                fresh_anchor = (plan.setpoints[plan.committed - 1]
                                if plan.committed else None)
                self._interval_execs.append(self._executor.execute_setpoints(
                    fresh, chunk_index=self._steps, on_tick=self._on_tick,
                    anchor=fresh_anchor))
            executions = self._interval_execs
            geo_mid = self._geo_after_commit
            d_ticks = int(plan.committed)
        self._steps += 1

        # One geodesic query per policy step (plus, under RTC with d > 0, the one at the
        # splice): the task layer keeps path length on the cheap PathTracker precisely so
        # stepping never triggers extra navmesh queries.
        self._task.observe_pose(self._robot_sim.get_2d_pose())
        geodesic = float(self._task.evaluate()["distance_to_goal"])

        if geo_mid is None:
            progress = (self._prev_geodesic - geodesic
                        if np.isfinite(geodesic) and np.isfinite(self._prev_geodesic) else 0.0)
            if self.progress_reward_clip > 0:
                progress = float(np.clip(progress, -self.progress_reward_clip,
                                         self.progress_reward_clip))
            r_commit = 0.0
        else:
            # The reward split (docs/RTC_RL.md section 4): progress over the committed
            # ticks vs the fresh span, each clipped by the same physical-consistency
            # bound, each finite-guarded the same way the single-delta path is. The two
            # parts telescope to the whole interval's progress; r_commit is what the
            # trainer-side re-timing moves to the action that committed it.
            r_commit = (base_geo - geo_mid
                        if np.isfinite(geo_mid) and np.isfinite(base_geo) else 0.0)
            eff_mid = geo_mid if np.isfinite(geo_mid) else base_geo
            r_fresh_progress = (eff_mid - geodesic
                                if np.isfinite(geodesic) and np.isfinite(eff_mid) else 0.0)
            if self.progress_reward_clip > 0:
                # PROPORTIONAL clips (audit note): the clip bound is physical -- no
                # part moves the geodesic more than the path drivable in ITS ticks --
                # so each part's bound scales with its tick share. Two full-size clips
                # would double the admissible artifact per interval at d > 0; at d = 0
                # this branch is not reached and the legacy single clip applies.
                clip_c = self.progress_reward_clip * d_ticks / self.gap
                clip_f = self.progress_reward_clip * (self.gap - d_ticks) / self.gap
                r_commit = float(np.clip(r_commit, -clip_c, clip_c))
                r_fresh_progress = float(np.clip(r_fresh_progress, -clip_f, clip_f))
            progress = r_commit + r_fresh_progress
        collided = any(self._collided(e) for e in executions)
        last_exec = next((e for e in reversed(executions) if e.ticks), None)
        reward = float(progress) - self.slack_penalty
        if collided:
            reward -= self.collision_penalty
        # Hold the last FINITE distance across snap dropouts: updating with inf would
        # zero the progress on the resumption step too, silently unpaying (and
        # unpunishing) all motion during the blind window. With the hold, the resumption
        # step settles the accumulated delta, bounded by progress_reward_clip.
        if np.isfinite(geodesic):
            self._prev_geodesic = geodesic
            self._blind_run = 0
        else:
            self._blind_run += 1

        if np.isfinite(geodesic) and geodesic < self._min_geodesic:
            self._min_geodesic = float(geodesic)
            self._path_at_min = float(self._task.path_tracker.length)

        reached = bool(np.isfinite(geodesic) and geodesic <= self.success_distance)
        # ESCAPE IS PHYSICAL, never a geodesic proxy: the SFT rejection rule (sustained
        # free-fall of joint_z, drive_failure.py thresholds), accumulated per tick in
        # _video_on_tick. A non-finite geodesic mid-episode is a navmesh-snap dropout --
        # the robot is standing in the world with the metric momentarily blind (measured:
        # under the old `escaped = not isfinite` rule, 42% of "escapes" were spawn-snap
        # failures and an unknown share of the rest transient) -- so it costs a
        # zero-progress step and the episode CONTINUES; success simply cannot fire while
        # the metric is blind.
        escaped = bool(self._fall_run >= self._fall_cfg.fall_duration_s / self.dt)
        # PERMANENT blindness (not a dropout): the metric never came back, so nothing in
        # this episode is learnable -- progress is structurally 0 and success cannot fire.
        # End it rather than spend the budget collecting a flat all-negative trajectory
        # that teaches "this state costs slack forever". Counted as truncation, not
        # termination: the episode is unmeasurable, not absorbing.
        reward_lost = bool(self.reward_lost_steps
                           and self._blind_run >= self.reward_lost_steps)
        done = bool(reached or escaped or reward_lost or self._steps >= self.max_steps)
        # Budget-cap ending is TRUNCATION, not termination: the MDP continues, the
        # episode doesn't. Consumers (GAE truncation bootstrap) treat it differently
        # from reached/escaped, which are genuinely absorbing.
        truncated = bool(done and not reached and not escaped)
        if reached and self.success_reward:
            reward += self.success_reward
        if escaped and self.escape_penalty:
            reward -= self.escape_penalty
        # Emitted EVERY step, not only the terminal one: `_pack_trajectory` takes an
        # episode's column set from its FIRST step's info, so terminal-only keys give a
        # length-1 episode (instant escape / near-spawn goal at a tight success_distance)
        # extra columns and `collate_trajectories` KeyErrors on the mixed batch (observed:
        # first v2 cycle). Values are running statistics; on the terminal step they equal
        # the old terminal-only emission exactly, so last-info-wins consumers are
        # unchanged. `ospl_fix` is the RECOMPUTED oracle SPL -- the task layer's own
        # OracleSPL is documented broken (docs/SAMPLE101_EVALS.md).
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
            "truncated": truncated,
            # Longest sustained fall so far, in seconds -- the escape detector's own
            # statistic (drive_failure fall_run_s), so near-misses are visible even when
            # no escape fires.
            "fall_run_s": float(self._max_fall_run * self.dt),
            # Blind-run length in policy steps, and whether it ended the episode. Emitted
            # every step so the screening gap is visible as a rate, not just at the end.
            "blind_run": int(self._blind_run),
            "reward_lost": reward_lost,
            "pos_rots": self._pos_rots(),
            **info_extra,
            # The reward split at the splice point (docs/RTC_RL.md section 4), emitted
            # EVERY step (0.0 / full reward without RTC) so the trajectory column set is
            # constant. r_commit is progress the COMMITMENT made -- caused by the
            # previous action's tail -- and is what the trainer-side re-timing moves;
            # r_fresh is everything else this step (fresh-span progress and all
            # penalties/bonuses; penalty timing within the interval is not split, a
            # recorded approximation).
            "r_commit": float(r_commit),
            "r_fresh": float(reward - r_commit),
            # How far the chunk asked the base to move this step. Paired with the
            # reward it separates "the policy commanded nothing" from "the policy
            # commanded a move the controller could not execute".
            "commanded_m": float(last_exec.ticks[-1].commanded_displacement)
            if last_exec is not None else 0.0,
            "end_lag_m": float(last_exec.end_lag) if last_exec is not None else 0.0,
        }
        self._cache["info"].append(info)
        self._cache["reward"].append(float(reward))
        obs: Dict[str, Any] = {"instr_or_goal": self._episode.object_category}
        if self._rtc:
            # Close this decision and open the next: the scheduler draws d at
            # OBSERVATION emission (the reciprocal schedule) and the prefix crosses to
            # the policy with the obs. On a terminal step nothing is pending -- emit an
            # empty prefix so the column set stays constant without leaking an
            # unconsumed observe() into the next episode.
            self._commit_done = False
            self._geo_after_commit = None
            self._interval_execs = []
            if done:
                obs["rtc_prefix"] = np.zeros((0, 3), dtype=np.float64)
                obs["rtc_delay"] = 0
            else:
                pfx, d_next = self._scheduler.observe(self._robot_sim.get_2d_pose())
                obs["rtc_prefix"] = pfx
                obs["rtc_delay"] = int(d_next)
        return rgb, {
            "obs": obs,
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
        # AFTER `_select`, deliberately. `_select` RAISES on a shard uid it cannot resolve,
        # so filtering first would turn a blacklisted-but-listed uid into a hard crash
        # instead of a quiet exclusion. Filtering here also keeps shard assignment itself
        # untouched: shards stay whatever the driver dealt, they just serve fewer episodes.
        if self._exclude_categories:
            before = len(episodes)
            episodes = [
                e for e in episodes
                if (getattr(e, "object_category", None) or "").strip().lower()
                not in self._exclude_categories
            ]
            dropped = before - len(episodes)
            if not episodes:
                raise ValueError(
                    f"exclude_categories={sorted(self._exclude_categories)} removed all "
                    f"{before} episodes from this actor's pool ({self.episodes_path}). "
                    "An empty pool would silently read as an exhausted shard."
                )
            print(f"[env] exclude_categories={sorted(self._exclude_categories)}: "
                  f"dropped {dropped}/{before} episodes, {len(episodes)} remain",
                  flush=True)
        self._episodes = episodes
        self._reshuffle()

    def _reshuffle(self) -> None:
        """FLAT permutation over the shard, freshly drawn each pass -- the discrete env's
        `shuffle=True` iterator semantics. No scene grouping: grouping trades sampling
        uniformity for simulator-rebuild savings, and the rollout scheduler already
        amortizes those. `self._rng` persists across passes, so pass k+1 is a NEW
        permutation (an RL actor must not replay the identical order every cycle), while an
        explicit `seed` still reproduces the whole stream."""
        idx = list(range(len(self._episodes)))
        # Training-stream restriction (see __init__). `_shard is None` IS the training
        # case; an assigned shard is an eval pass and must serve exactly what it was
        # given, so the filter is skipped there.
        if self._train_uids is not None and self._shard is None:
            keep = [i for i in idx if self._episodes[i].uid in self._train_uids]
            if not keep:
                raise ValueError(
                    f"train_uids matched none of this actor's {len(idx)} episodes; "
                    "an empty training order would read as an exhausted shard")
            # Printed on EVERY reshuffle, not only on a mismatch. A held-out run's whole
            # claim is that training never sees the eval episodes, and silence is not
            # evidence of that -- it reads identically to a filter that never ran.
            print(f"[env] train_uids: serving {len(keep)} of this actor's {len(idx)} "
                  f"episodes ({len(self._train_uids)} uids requested)", flush=True)
            idx = keep
        self._order = [int(i) for i in self._rng.permutation(len(idx))]
        self._order = [idx[i] for i in self._order]
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
            if not self.screen_reachability:
                # Cross-path parity mode: serve EVERYTHING, like the harness's default.
                # The harness scores all of sample101 on the dataset mesh; this env's
                # robot-footprint screen drops ~30 of those 101 as goal_unreachable,
                # which is the right call for TRAINING (below) and the wrong one for a
                # comparison whose reference includes them as (almost certain) failures.
                return episode, "unscreened"
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
        """Non-finite distances become NaN -- a FLOAT, never None.

        History (2026-08-24 audit, F1): this used to return None, and one None in a
        numeric info column made `_pack_trajectory` build an object-dtype array that
        `collate_trajectories` could not convert -- its old bare `except` then dropped
        the ENTIRE column for the whole batch, silently starving every
        distance-consuming consumer (~0.16% of steps poisoned ~10% of episodes'
        batches). NaN keeps the column numeric end to end and is maskable downstream;
        JSON strictness (null, not NaN) is handled at serialisation time
        (`_denan` in flush_logs_to_disk), which is the only place it was ever needed."""
        return float(value) if np.isfinite(value) else float("nan")
