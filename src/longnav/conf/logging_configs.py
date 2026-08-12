from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

cs = ConfigStore.instance()


# --- logger backend configs ---
@dataclass
class WandbLoggerConfig:
    _target_: str = "longnav.utils.logging_workers.WandbLoggerActor"
    project: str = MISSING


cs.store(name="wandb", group="logger", node=WandbLoggerConfig())
