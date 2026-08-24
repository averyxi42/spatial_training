from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()


# --- habitat sim config ---
@dataclass
class HabitatEnvConfig:
    _target_: str = "longnav.env.habitat.HabitatEnvActor"
    config_path: str = "habitat_configs/objectnav_hm3d_rgbd_semantic.yaml"
    dataset_path: Optional[str] = None
    workspace: Optional[str] = "."
    scenes_dir: Optional[str] = None
    split: str = "val"
    fp_guard: bool = False
    fn_guard: bool = False
    voxel_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: None)
    output_schema: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "obs": {"rgb": True, "instr_or_goal": True, "patch_coords": False},
        "info": {"episode_label": True, "spl": True, "soft_spl":True, "success": True,"distance_to_goal":True},
        "done": True,
        "reward": True,
        "stuck": True,
        "fp_stop": True
    })
    auto_flush: bool = False # automatically flush logs upon reset
    ep_seed: Optional[bool] = None # if set, episode iterators are deterministic with same set seed all habitat workers
    explr_bonus: Optional[float] = 0.13
    collision_penalty: Optional[float] = 0.05
    fpstop_penalty: Optional[float] = 0.3
    add_top_down_map: bool = False
    per_worker_config_overrides: Optional[List[str]] = None


# --- dummy sim configs ---
@dataclass
class DummyDiscreteEnvConfig:
    _target_: str = "longnav.env.env_base.DummyEnvActor"


@dataclass
class DummyContinuousEnvConfig:
    _target_: str = "longnav.env.env_base.DummyContinuousEnvActor"


@dataclass
class ContinuousObjectNavEnvConfig:
    """ObjectNav driven by action chunks through the PID tracker. See docs/LATENT_RL_ENV.md.

    One env step is one chunk: the first `gap` setpoints are tracked, one control tick each,
    and the tail is discarded exactly as training and the eval harness do. `gap * dt` seconds
    of simulated time per step -- the same currency the eval budget is quoted in.
    """

    _target_: str = "longnav.env.objectnav_continuous.ContinuousObjectNavEnvActor"
    episodes: str = "/Projects/spatial_training/data/datasets/objectnav/hm3d/v1/val"
    scene_root: Optional[str] = "/Projects/spatial_training/data/scene_datasets"
    episode_source: str = "objectnav"
    gap: int = 10
    dt: float = 0.04
    max_steps: int = 175          # 70 s at gap 10, the budget every sample101 number uses
    success_distance: float = 1.0
    # `dataset`, not `robot`: on the robot mesh `snap_point` can resolve the proxy across an
    # island boundary and the metric silently starts measuring a different goal instance.
    navmesh: str = "dataset"
    distance_to: str = "VIEW_POINTS"
    sensor_uuid: str = "color_sensor"
    width: int = 640
    height: int = 480
    pid_preset: str = "baseline"
    # Reward shaping. Progress is the geodesic reduction per step and needs no coefficient;
    # these two are the costs. Collision is measured on OBSTRUCTION contacts, never the raw
    # count -- the robot is always touching the floor.
    slack_penalty: float = 0.0
    collision_penalty: float = 0.0
    # Clip on the per-step progress REWARD (never the metrics): navmesh snap of a physical
    # robot occasionally relocates across a wall/floor, injecting multi-metre single-step
    # geodesic jumps (measured 0.16% of steps, max 6.7 m). No physical step moves the
    # geodesic more than the path driven in gap*dt (~0.6 m), so beyond this is artifact.
    # 0 disables.
    progress_reward_clip: float = 0.75
    # Terminal bonus added to the reached step's reward (discrete-standard success term).
    # 0 keeps the progress-only shape every run before 2026-08-21 used.
    success_reward: float = 0.0
    # Subtracted from the escaped terminal step's reward. Unpenalized escapes end the
    # episode keeping all accumulated progress -- a free exit the policy drifts toward
    # (measured: escape rate doubled over the sd04_noterm run). 0 keeps the old shape.
    escape_penalty: float = 0.0
    # End an episode after N consecutive policy steps with a NON-FINITE geodesic. DOA
    # screening runs at the SPAWN SNAP, but an episode can lose the reward metric
    # permanently once the robot moves (measured: 1S7LAXRdDqK:48 -- finite 6.89 m at
    # spawn, blind thereafter; the finite-hold then froze progress at ~0 across 175
    # steps and 44 m of wandering, 0/29 in every run that served it). Such an episode is
    # unmeasurable, not hard: it contributes a flat all-negative trajectory and silently
    # caps the pool's reachable success. Transient snap dropouts last ticks, so 25 steps
    # (10 s of sim) sits far above their scale. 0 disables (pre-2026-08-14 behaviour).
    reward_lost_steps: int = 25
    # Goal categories dropped from the pool entirely, e.g. ["plant"]. EMPTY BY DEFAULT.
    # `plant` in particular is often mislabelled in HM3D, so the policy is charged for
    # failing to find something that is not there.
    #
    # Interactions, all deliberate:
    # * applied AFTER shard resolution, so a shard uid naming an excluded episode is
    #   still resolvable (`_select` RAISES on unresolved uids) -- it is dropped, not fatal;
    # * `list_episode_uids` therefore reports the FILTERED pool, so `build_eval_partition`
    #   draws its fixed eval set from it too. Train and eval stay consistent within a run,
    #   but two runs with different exclusions DO NOT share an eval set and their numbers
    #   are not comparable;
    # * orthogonal to reachability screening (`_next_admissible_episode`): excluded
    #   episodes never reach the screener, so the `skipped_so_far` reason codes keep
    #   counting genuine navmesh failures only;
    # * emptying a pool raises rather than reading as an exhausted shard.
    exclude_categories: Optional[List[str]] = None
    # Restrict the TRAINING episode stream to these uids (a list, or a path to a file of
    # comma/newline-separated uids). Empty = the whole filtered pool, which is every run
    # before 2026-08-15. Applied in `_reshuffle` and ONLY when no shard is assigned, so
    # `run_eval_cycle` -- which assigns explicit eval shards and restores `assign_shard(None)`
    # -- still reaches episodes training never serves. That is the point: with
    # `task.eval_uids_file` naming a DISJOINT set, the in-training eval stops being
    # quasi-held-out (the pool's eval episodes are trained on every len(pool)/n_rollout
    # cycles) and becomes a real generalisation signal.
    #
    # Uids are `scene:episode_id[#occurrence]` and the occurrence counter is assigned per
    # shard FILE, so a uid is only meaningful against the pool it was written from. Restrict
    # the stream over the full pool rather than building a derived dataset -- a derived
    # subset renumbers occurrences and silently invalidates every pinned uid.
    train_uids: Optional[Any] = None
    # None (the RL default) = a distinct per-worker seed, the discrete env's ep_seed
    # pattern; set an explicit int to reproduce an exact episode stream (eval).
    seed: Optional[int] = None
    source_kwargs: Optional[Dict[str, Any]] = None
    # False = per-episode MP4 + thumbnail + sequence.json through the amortized
    # flush_logs_to_disk hook; True skips the video encode and writes scalars only.
    minimal_logging: bool = False
    # Video temporal resolution: capture a frame every N physics ticks (1/dt = 25 Hz base,
    # so stride 1 = 25 fps of sim time, stride 2 = 12.5 fps, stride 10 = one frame per
    # policy step -- the old choppiness). Frames are JPEG-encoded in memory as captured.
    # Default 2: smooth to the eye at half the render/encode/storage cost of stride 1.
    video_tick_stride: int = 2
    # Playback speed relative to sim time: written fps = factor / (dt * stride), so the
    # default 1.0 plays exactly realtime at ANY stride; 2.0 twice realtime, 0.5 half.
    video_realtime_factor: float = 1.0
    # -- RTC latency masking (docs/RTC_RL.md; the schedule is objectnav_eval's) --------
    # "none" (default) is the historical env, bit for bit. Any other source turns on the
    # reciprocal schedule: the head sends the FULL chunk (requires an RTC-trained
    # checkpoint -- fm_config.rtc_delay_max > 0), the env owns the slicing, the obs
    # carries the committed prefix, and info reports the r_commit/r_fresh reward split.
    # d is ALWAYS the assumed delay -- configured/sampled, never wall-clock.
    #   fixed: rtc_delay every decision (the eval sweep's knob)
    #   uniform: d ~ U[0, rtc_delay_max]
    #   exp: d ~ rtc_delay_base**d over [0, rtc_delay_max] -- the RL default, matching
    #        the RTC checkpoints' own training law
    rtc_delay_source: str = "none"
    rtc_delay: int = 0
    rtc_delay_max: int = 0
    rtc_delay_base: float = 0.8
    # Separate stream (seeded rtc_delay_seed + episode index): delay draws must not
    # perturb the policy's own noise, or no difference between delay policies is
    # attributable.
    rtc_delay_seed: int = 0
    # Injected chunk-toss probability per decision (fallback A: hold). Deployment-time
    # disturbance only; keep 0 during training.
    rtc_overrun_rate: float = 0.0


@dataclass
class ColorBanditEnvConfig:
    _target_: str = "longnav.env.color_bandit.ColorBanditEnvActor"


@dataclass
class ReplayEnvConfig:
    _target_: str = "longnav.env.replay.ReplayEnvActor"
    # A scripted sequence of {"rgb": ndarray, "obs": {...}, "reward": float,
    # "done": bool, "info": {...}} entries. Left None here -- ndarrays aren't
    # something a static Hydra config should carry -- tests pass a real
    # script via hydra.utils.instantiate(cfg.sim, script=[...]), overriding
    # this field at instantiation time.
    script: Optional[List[Dict[str, Any]]] = None


cs.store(name="habitat", group="sim", node=HabitatEnvConfig())
cs.store(name="voxel", group="sim", node=HabitatEnvConfig(
    voxel_kwargs={
        "patch_size": 32,
        "resolution": 0.15,
        "fov_degrees": 79
    }, #set to none for standard mode
    output_schema={
        "obs": {"rgb": True, "instr_or_goal": True, "patch_coords": False},
        "info": {"episode_label": True, "spl": True, "success": True},
        "done": True,
    }
))
cs.store(name="dummy_discrete", group="sim", node=DummyDiscreteEnvConfig())
cs.store(name="dummy_continuous", group="sim", node=DummyContinuousEnvConfig())
cs.store(name="objectnav_continuous", group="sim", node=ContinuousObjectNavEnvConfig())
cs.store(name="color_bandit", group="sim", node=ColorBanditEnvConfig())
cs.store(name="replay", group="sim", node=ReplayEnvConfig())
