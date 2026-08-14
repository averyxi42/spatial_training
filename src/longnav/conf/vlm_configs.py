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
    # Exploration temperature: sigma_rl = tau * sigma_phi (docs/LATENT_RL.md "At RL time").
    # 1.0 = the checkpoint's own trained sigma -- the evidenced setting: the pilot's vary-c
    # attribution (82% of within-episode return variance, best-of-3 0.611 -> 0.875) was
    # measured at exactly that scale. Sweep only against a behavioral-spread measurement;
    # the tau response is sub-linear (10x tau bought 5.3x spread on the pilot).
    tau: float = 1.0
    # DIAGNOSTIC ONLY, and None is correct. The action is `c`; the decoder and the physics
    # are the environment, so `z_0` is environment noise and the PPO ratio over `c` is exact
    # without touching it. Pinning selects one arbitrary slice of the policy -- measured at
    # 2.4x the ensemble's mean path length -- so it belongs in probes, not in rollouts.
    pin_flow_noise_seed: Optional[int] = None
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
class FlowSDEHeadConfig:
    """The denoising chain as the action. See docs/FLOW_SDE_RL.md.

    `type: continuous` for the same reason the latent head's is: the branch sites dispatch
    on head CAPABILITY (`sample_chain_np` / `chain_log_prob_batch`), never on a type string,
    so the validator at vlm_worker.py:59, the wrapper and every existing path are untouched.
    The natural checkpoint is the DETERMINISTIC one -- RL initialises at the strongest SFT
    policy; that asymmetry over the latent head is the point of the design.
    """

    _target_: str = "longnav.utils.flow_sde_policy.FlowSDEHead"
    type: str = "continuous"
    # The SFT checkpoint carrying the readout MLP and the velocity field. No default.
    checkpoint_dir: Optional[str] = None
    # Ticks executed per policy step; the chunk tail is discarded as everywhere else.
    gap: int = 10
    # Stochastic denoising steps per chunk, of the checkpoint's K (=num_inference_steps).
    # 3 for the first runs: exploration at low chain-ratio variance. Sweeping n at fixed lr
    # measures step size, not exploration -- scale lr ~ 1/n (docs/FLOW_SDE_RL.md).
    sde_n: int = 3
    # THE exploration scale, sigma_t = a * sqrt(t/(1-t)). No default by design: its usable
    # range is bounded above by fidelity and below by the checkpoint's own z_0 scatter
    # (0.737 rad on cotrain-v3), and it must come from the noise sweep.
    sde_noise_a: Optional[float] = None
    # Keep stochastic positions away from the 1/t score singularity.
    sde_exclude_last: int = 1
    # "none" | "sigma": per-transition log-prob weight sigma_k/sigma_0. The on-policy
    # density gradient scales as 1/sigma_k, so unweighted, late low-noise refinement
    # steps dominate every update ~10x (measured: decoder grad x1.0 -> x9.6 across
    # positions); "sigma" equalizes per-position gradient scale at 0.914 direction
    # overlap. A deliberate biased surrogate in the logprob_reduction:"mean" tradition,
    # applied identically in sampler and scorer -- see SDEConfig.position_weight.
    sde_position_weight: str = "none"
    # Seeds the head's private sampling stream; None = fresh draws (normal training).
    sde_seed: Optional[int] = None
    # NO CLIPPING, same reason as the latent head: the "action" is the 660-float chain and
    # clipping its latents is meaningless.
    continuous_action_clip_low: float = float("-inf")
    continuous_action_clip_high: float = float("inf")
    # entropy_bonus MUST stay null in the algo config: chain entropy is fixed by the
    # sigma_t schedule and rl_loss raises rather than silently adding a zero-gradient
    # constant. Present-but-unused keys below keep Gaussian-head config lookups from
    # KeyError, as on the latent head.
    logprob_reduction: str = "sum"
    gaussian_init_log_std: float = -0.5
    gaussian_min_log_std: float = -20.0
    gaussian_max_log_std: float = 2.0


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
    # "regression" (clipped-MSE ValueHead, the pre-existing head) or "distributional"
    # (DistributionalValueHead: categorical over [v_min, v_max], HL-Gauss cross-entropy --
    # bounded, scale-free critic gradients; cliprange_value is unused there). The fields
    # below only apply to "distributional".
    kind: str = "regression"
    n_bins: int = 51
    # Support of the categorical critic. Pick generously from the reward shape: progress
    # telescopes to ~start_m (p90 ~12 m discounted down by gamma), plus success_reward
    # and minus escape_penalty. Out-of-range returns clamp to the edge bins.
    v_min: float = -5.0
    v_max: float = 15.0
    hl_sigma_ratio: float = 0.75   # target smearing, in bin widths (HL-Gauss standard)
    # Critic readout position, relative to the POLICY readout token (the sandwich `**`).
    # 0 = share the policy's token (historical behavior). Negative offsets land in the
    # "Action:" text immediately before the assistant prefix -- text tokens, so they
    # survive both the turn crop and the sparsifier, with no template/tokenizer change.
    # A separate token removes the first-order gradient collision at the one activation
    # the flow policy hangs from (h drift is 1/sigma^2-amplified into policy movement);
    # second-order interference through shared WEIGHTS remains -- that is what SFT
    # value co-training addresses, and this offset is part of that checkpoint contract.
    readout_offset: int = 0


cs.store(name="lm_head", group="policy_head", node=LMHeadConfig())
cs.store(name="gaussian_head", group="policy_head", node=GaussianHeadConfig())
cs.store(name="latent_head", group="policy_head", node=LatentHeadConfig())
cs.store(name="flow_sde_head", group="policy_head", node=FlowSDEHeadConfig())
