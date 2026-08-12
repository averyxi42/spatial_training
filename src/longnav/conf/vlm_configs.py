from dataclasses import dataclass, field
from typing import List, Optional

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
    # Optional: seed torch's global RNG immediately before constructing the
    # head's mean layer (see VLMWorker._ensure_continuous_action_head).
    # Without a loaded checkpoint, the mean layer's nn.Linear init draws
    # from the unseeded global torch RNG, so its initial weights (and
    # therefore mu/log-prob) differ across every process launch -- this is
    # normal/desired for real training (independent random init per run),
    # but makes deterministic fingerprint tests of an untrained continuous
    # head impossible without pinning it. None (default) preserves the
    # existing unseeded behavior.
    action_head_init_seed: Optional[int] = None


@dataclass
class LatentHeadConfig:
    """The CVAE prior as the policy. See docs/LATENT_RL_ENV.md.

    `type: continuous` deliberately -- this reuses the whole existing continuous path
    (sampling, log-prob, storage, ratio) unchanged; only where mu/log_std come from and what
    the env receives differ.
    """

    _target_: str = "longnav.utils.latent_policy.LatentIntentHead"
    type: str = "continuous"
    # The SFT checkpoint carrying the split, the readout MLP and the velocity field. There is
    # no useful default: a freshly initialised latent head is noise with the right shape.
    checkpoint_dir: Optional[str] = None
    # `c` is the action, so this must equal the checkpoint's fm_context_dim; disagreeing is
    # an error rather than a reshape.
    action_space_dim: int = 1024
    # Ticks executed per policy step. The rest of the chunk is discarded, as in training and
    # in the eval executor -- the remainder is superseded by the next observation.
    gap: int = 10
    # Freezes the ODE base noise so `c -> chunk` is deterministic and differentiable. Required
    # (the head raises without it): otherwise the same `c` decodes differently every call and
    # `old_log_prob` describes an action that never executed.
    pin_flow_noise_seed: Optional[int] = 0
    # NO CLIPPING. The Gaussian head's +/-1.0 is right for a 2-d control vector and wrong for
    # a 1024-d latent: it would clamp `c`, and the log-prob would then be evaluated at the
    # clipped action under the unclipped Gaussian.
    continuous_action_clip_low: float = float("-inf")
    continuous_action_clip_high: float = float("inf")
    # Far below the Gaussian head's -5.0. The ELBO fits sigma near 1% of h's per-dim std
    # (log_std ~ -6.9); the tighter floor would clamp it and inflate exploration ~500x.
    gaussian_min_log_std: float = -20.0
    gaussian_max_log_std: float = 2.0
    # `sum` is the joint log-density and the default everywhere. `mean` is the
    # dimension-level analogue of token-level RLHF ratios, for when 1024 dims saturate the
    # PPO clip -- a deliberate bias, never a silent one. See reduce_gaussian_logprob.
    logprob_reduction: str = "sum"
    # Unused by this head (sigma is trained, not initialised) but present so a config written
    # against the Gaussian head does not KeyError on lookup.
    gaussian_init_log_std: float = -0.5


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
cs.store(name="latent_head", group="policy_head", node=LatentHeadConfig())
