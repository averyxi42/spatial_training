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
cs.store(name="color_bandit", group="sim", node=ColorBanditEnvConfig())
cs.store(name="replay", group="sim", node=ReplayEnvConfig())
