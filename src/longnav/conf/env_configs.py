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
    # Extra decisions after first success are normally supervised only by the stop head.
    post_goal_steps: int = 0
    # Optional smooth reward for a stationary decoded chunk while in the post-goal tail.
    # It is disabled by default and paired with rollout.learn_post_goal_actions.
    post_goal_stillness_reward: float = 0.0
    post_goal_stillness_scale_m: float = 0.1
    # Rewards attached to an explicit policy STOP.  They are disabled by default so
    # legacy binary-head rollouts preserve their return exactly.
    policy_stop_correct_reward: float = 0.0
    policy_stop_false_penalty: float = 0.0
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
    # Environment-only RGB-D BEV reward. The VLM receives RGB exactly as before.
    exploration_enabled: bool = False
    exploration_reward_weight: float = 0.0
    exploration_reward_sigma_m2: float = 1.0
    exploration_resolution_m: float = 0.1
    exploration_local_window_m: float = 24.0
    exploration_ray_range_m: float = 12.0
    exploration_n_rays: int = 31
    exploration_path_samples: int = 4
    exploration_max_depth_m: float = 12.0
    exploration_depth_sensor_uuid: str = "exploration_depth"


@dataclass
class NavVerseEnvConfig:
    """NavVerse (Isaac Lab) continuous ObjectNav/PlaceNav, driven by SE(2) chunks through
    NavVerse's own velocity waypoint follower. See `longnav/env/navverse.py`.

    One env step is one chunk of `gap` cumulative body-frame SE(2) setpoints, executed as a
    single TIMED_TRAJECTORY -- the same mechanism and convention (relative_se2_to_world) the
    continuous_flow benchmark backend drives over the wire, so training and eval execute
    chunks identically.
    """

    _target_: str = "longnav.env.navverse.NavVerseEnvActor"
    # NavVerse resolves its own config/episode/scene paths relative to CWD; a Ray actor
    # inherits this trainer's CWD, not the main NavVerse-Benchmark repo's, so the actor
    # chdir()s here first.
    navverse_repo_root: str = "/home/ubuntu/Projects/NavVerse-Benchmark"
    config_path: str = "configs/default.yaml"
    episode_folder: str = "/home/ubuntu/Projects/navverse_data/episodes/"
    scene_folder: str = "/home/ubuntu/Projects/navverse_data/"
    # One episode label per line -- `episodes/train_set.txt` / `test_set.txt` / etc in the
    # main NavVerse-Benchmark repo. Required whenever `assign_shard(None)` can happen (the
    # framework's trivial-shard / shard_size=0 training default): the env owns its dataset
    # scope, so "everything" means everything in THIS file, not everything VLNSim happens
    # to have loaded from `episode_folder` -- otherwise train/test separation would depend
    # on episode_folder's contents rather than being asserted here.
    episodes_path: Optional[str] = None
    train_uids: Optional[Any] = None
    excluded_episode_labels: Optional[List[str]] = None
    task_type: str = "placenav"
    robot_name: str = "spot"
    tidybot_embodiment: str = "full"
    hide_robot_visuals: bool = False
    on_demand_render: bool = False
    camera_rgb_only: bool = False
    policy_camera_only: bool = False
    disable_camera: bool = False
    trajectory_planner: Optional[str] = None
    profile_step_timing: bool = False
    profile_step_interval: int = 100
    gap: int = 10
    dt: float = 0.04
    max_steps: int = 175
    # NavVerse's own benchmark convention (navverse_tools/statistics/analysis_legacy_vln_success16.py),
    # not habitat's 1.0/0.2 -- keeps train and the project's own eval protocol consistent.
    success_distance: float = 1.6
    slack_penalty: float = 0.0
    collision_penalty: float = 0.0
    failure_penalty: float = 0.0
    progress_reward_clip: float = 0.75
    success_reward: float = 0.0
    timeout_margin_s: float = 1.0
    minimal_logging: bool = False
    video_fps: int = 2
    # Diagnostic capture for NavVerse: store one frame every N low-level physics ticks.
    # Zero preserves the ordinary policy-step-only logging path.
    video_tick_stride: int = 0
    video_realtime_factor: float = 1.0


@dataclass
class NavVerseProxyBatchedEnvConfig:
    """Batched variant of `navverse`: `num_hosts` Isaac Lab processes, each running
    `slots_per_host` vectorized robots, presented to `rollout_core` as
    `num_hosts * slots_per_host` ordinary-looking single-episode actors
    (`longnav.env.navverse.NavVerseSlotProxyActor`). Recovers the pre-flowsde vectorized
    env's amortization of Isaac Lab's ~60GB/process fixed cost, at the price of lockstep
    coupling between the `slots_per_host` siblings of one host -- see the module docstring
    in `longnav/env/navverse.py` before changing `slots_per_host`.

    `resources.num_sims` MUST equal `num_hosts * slots_per_host`, since `SimWorkerFactory`
    constructs exactly that many proxy actors and each one claims the next sequential slot.
    `episodes_path` must group into scenes of >= `slots_per_host` episodes each (a host's
    whole batch shares one scene per round) -- this repo's smoke sample (16 scenes x 8
    episodes) is sized for the default `slots_per_host=8`.
    """

    _target_: str = "longnav.env.navverse.NavVerseSlotProxyActor"
    num_hosts: int = 8
    slots_per_host: int = 8
    # Fractional GPU request for each HOST actor (the proxies themselves request ~0 GPU
    # through `resources.sim_gpu_fraction`, set separately in the resources/*.yaml). Ray's
    # fractional-GPU scheduling bin-packs by request order but does not hard-pin a host and
    # its paired VLM worker to one physical device -- verify actual placement (e.g.
    # `ray.get_gpu_ids()` inside the host, or `nvidia-smi` during the smoke run) before
    # trusting colocation at full scale.
    host_num_gpus: float = 0.5
    host_num_cpus: int = 4
    host_conda_env: Optional[str] = None
    navverse_repo_root: str = "/home/ubuntu/Projects/NavVerse-Benchmark"
    config_path: str = "configs/default.yaml"
    episode_folder: str = "/home/ubuntu/Projects/navverse_data/episodes/"
    scene_folder: str = "/home/ubuntu/Projects/navverse_data/"
    episodes_path: Optional[str] = None
    train_uids: Optional[Any] = None
    excluded_episode_labels: Optional[List[str]] = None
    task_type: str = "placenav"
    robot_name: str = "spot"
    tidybot_embodiment: str = "full"
    hide_robot_visuals: bool = False
    on_demand_render: bool = False
    camera_rgb_only: bool = False
    policy_camera_only: bool = False
    disable_camera: bool = False
    trajectory_planner: Optional[str] = None
    profile_step_timing: bool = False
    profile_step_interval: int = 100
    gap: int = 10
    dt: float = 0.04
    max_steps: int = 175
    success_distance: float = 1.6
    slack_penalty: float = 0.0
    collision_penalty: float = 0.0
    failure_penalty: float = 0.0
    progress_reward_clip: float = 0.75
    success_reward: float = 0.0
    timeout_margin_s: float = 1.0
    minimal_logging: bool = False
    video_tick_stride: int = 0
    video_realtime_factor: float = 1.0


@dataclass
class NavVerseBatchedEnvConfig(NavVerseEnvConfig):
    """One Ray simulator actor owning a fixed batch of NavVerse robot environments."""

    _target_: str = "longnav.env.navverse.NavVerseHostActor"
    slots_per_host: int = 8


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
cs.store(name="navverse", group="sim", node=NavVerseEnvConfig())
cs.store(name="navverse_batched", group="sim", node=NavVerseBatchedEnvConfig())
cs.store(
    name="navverse_proxy_batched",
    group="sim",
    node=NavVerseProxyBatchedEnvConfig(),
)
cs.store(name="color_bandit", group="sim", node=ColorBanditEnvConfig())
cs.store(name="replay", group="sim", node=ReplayEnvConfig())
