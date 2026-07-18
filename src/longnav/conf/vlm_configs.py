from dataclasses import dataclass, field
from typing import List

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()


# --- policy head configs ---
@dataclass
class LMHeadConfig:
    """Discrete action head: the backbone's native LM output layer plus vocab-token parsing.
    No _target_ - there's nothing extra to construct beyond the backbone."""
    type: str = "discrete"
    vocab: List[str] = field(default_factory=lambda: ["stop", "forward", "left", "right"])


@dataclass
class GaussianHeadConfig:
    _target_: str = "longnav.utils.vlm_worker.ContinuousActionHead"
    type: str = "continuous"
    action_space_dim: int = 2
    gaussian_init_log_std: float = -0.5
    gaussian_min_log_std: float = -5.0
    gaussian_max_log_std: float = 2.0
    continuous_action_clip_low: float = -1.0
    continuous_action_clip_high: float = 1.0


# --- value head config (optional, auxiliary; no ConfigStore group needed - see plan) ---
@dataclass
class ValueHeadConfig:
    _target_: str = "longnav.utils.vlm_worker.ValueHead"
    learning_rate: float = 5e-4  # Often higher than Adapter LR
    dropout: float = 0.0
    dtype: str = "float32"
    # List of hidden layer sizes. Empty list [] implies a single linear layer (Linear Probe).
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 512])
    cliprange_value: float = 0.2
    value_grad_scale: float = 0.1


cs.store(name="lm_head", group="policy_head", node=LMHeadConfig())
cs.store(name="gaussian_head", group="policy_head", node=GaussianHeadConfig())
