from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from omegaconf import MISSING

# --- 1. Resource & Environment Config ---
@dataclass
class ResourceConfig:
    ray_address: str = "local"
    object_spilling_directory: str = "./ray_object_spilling"
    osm_gb: int = 128  # Object Store Memory in GB
    vlm_resource_tag: str = "env_a"
    sim_resource_tag: str = "env_b"
    master_addr: str = 'localhost'
    master_port: Optional[int] = None #port for accelerate/ddp
    num_vlms: int = 1
    num_sims: int = 1
    vlm_conda_env: Optional[str] = "longnav"
    habitat_conda_env: Optional[str] = "vln"
    vlm_gpu_fraction: float = 0.7
    sim_gpu_fraction: float = 0.14
    vlm_cpus: int = 4
    sim_cpus: int = 4

# --- 3. Model & Worker Configs ---
@dataclass
class VLMConfig:
    model_id: Optional[str] = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
    attn_impl: str = "flash_attention_2"
    dtype: str = "bfloat16"
    prefix: str = '<|im_start|>assistant\n**'
    postfix: str = '**<|im_end|>\n'
    policy_head: Any = MISSING
    offload_cache: bool = False
    use_sparse: bool = True
    save_outputs: bool = False # only need this for RL
    # SFT adapter dir to MERGE into the base at load (merge_and_unload), making the base
    # model the SFT policy and the trainable LoRA a fresh zero-delta on top. Required for
    # rl_config.ref_kl on continuous heads: it is what makes peft's stock
    # disable_adapter() an exact reference to the SFT policy. Mutually exclusive with
    # pointing training.checkpoint at the SAME adapter (that would apply the delta twice).
    merge_adapter_dir: Optional[str] = None

@dataclass 
class PolicyLossConfig:
    name: str = "vanilla"
    clip_cov_ratio: Optional[float] = None
    clip_cov_ub: Optional[float] = None
    clip_cov_lb: Optional[float] = None

@dataclass
class RLAlgoConfig:
    # generic on policy params
    value_head: Optional[Any] = None
    # Frozen SFT-cotrained state probe (value + distance heads over one readout
    # hidden). "auto" loads state_probe.pt from the policy head's checkpoint_dir when
    # present; a path loads that dir; None disables. Mutually exclusive with
    # value_head (both claim the worker's value readout slot).
    state_probe: Optional[str] = None
    advantage_estimator: str = "reinforce_plus_plus"
    # RTC return re-timing (docs/RTC_RL.md section 4): r~_k = r_fresh_k +
    # gamma*r_commit_{k+1} == R_k - r_commit_k -- subtract from each step's return the
    # committed reward its action could not influence. Consumed only when the env
    # emits the r_commit/r_fresh split (RTC runs); default ON per the doc's identity
    # argument, flag kept for the ablation.
    retime_commit_rewards: bool = True
    n_rollout: int = 12 # note: must be divisible by num vlms times gradient accumulation
    n_adv: int = 256 # number of trajectories for advantage estimation, must > n_rollout
    n_epoch: int = 2 # number of policy gradient epochs

    # PPO Hyperparameters
    clip_ratio: float = 0.2
    clip_ratio_low: Optional[float] = None
    clip_ratio_high: Optional[float] = None
    clip_ratio_c: float = 3.0
    loss_agg_mode: Optional[str] = "token-mean" #"seq-mean-token-sum" seq-mean-token-mean
    
    # clip cov parameters
    policy_loss:PolicyLossConfig = field(default_factory=PolicyLossConfig)
    # GAE Hyperparameters
    gamma: float = 0.99
    lam: float = 0.95

    time_kernel_sigma: float = 50.0
    time_alignment: str = "start" #or end
    time_loto: bool = False # leave one trajectory out
    
    distance_kernel_sigma: float = 0.5
    distance_clip_max: Optional[float] = 17.0
    distance_clip_percentile: Optional[float] = 0.95
    distance_pad_mode: Optional[str] = "replicate"
    distance_pad_val: Optional[float] = None
    # Value & Entropy. Optional so a chain-action head can set it to null explicitly --
    # chain entropy is schedule-fixed and rl_loss raises on a NONZERO bonus there.
    entropy_bonus: Optional[float] = 0.0
    # Scale each episode-minibatch's loss by (its token count / cycle mean): the
    # accumulated gradient becomes the global TOKEN mean instead of the episode mean.
    # Off = the historical episode-weighted objective every run to date used (measured
    # bias: quick successes get ~4x per-token influence; mean pg_loss offset -0.26).
    token_weighted_loss: bool = False
    # Bootstrap gamma*V(s_T) onto the last reward of TRUNCATED (budget-capped) episodes.
    # Off treats the cap as absorbing, under-crediting long episodes' tails. Requires a
    # value head and the env's `truncated` flag; silently inert without both.
    bootstrap_truncated: bool = False

    # Ref KL Control
    use_ref: bool = True
    kl_coeff: float = 0.001
    kl_target: float = 0.1

    # Continuous-head tether to the INIT policy (the frozen-head arm's ref-KL; discrete
    # keeps its own use_ref/kl_coeff path above, which references the BASE model via
    # disable_adapter -- a different and, for us, wrong reference). ref_kl=True makes
    # every training postprocess run one extra no-grad forward with the peft weights
    # swapped to their first-postprocess snapshot (== the loaded SFT/init policy) and
    # store log pi_ref at the stored actions; rl_loss then logs ref/kl_k1 and ref/kl_k3
    # every step. MEASURE, NOT CONSTRAIN, by default:
    ref_kl: bool = False
    # 0.0 = measure only (the default, deliberately -- read the gauge before trusting it
    # as a leash). >0 adds ref_kl_coeff * k3 to the policy loss (k3 = the non-negative
    # low-variance KL(pi||pi_ref) estimator, verl's low_var_kl convention).
    ref_kl_coeff: float = 0.0

    # # Compatibility for verl's agg_loss
    # @property
    # def global_batch_info(self):
    #     # For single-worker testing, batch size is 1
    #     return {}# "dp_size": 1, "global_batch_size": 1
    global_batch_info: Optional[Dict[str,Any]] = field(default_factory=lambda:{})
    # Helper to support config.get("key", default) used in loss functions
    def get(self, key, default=None):
        return getattr(self, key, default)

@dataclass
class SFTConfig:
    pass

# --- training configs ---
@dataclass
class HydraLoraConfig:
    """
    A Hydra-compatible mirror of peft.LoraConfig.
    Removes Union types (like str | List[str]) that crash OmegaConf.
    """
    r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.0
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    # Enforce List[str] to satisfy Hydra. 
    # If you need regex (str), you can change this to Any, but List is safer.
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # modules_to_save is also a list, defaulting to None is fine for Hydra
    modules_to_save: Optional[List[str]] = None
    use_rslora: bool = False #rank stabilized lora. should use?

@dataclass
class VLMTrainingConfig:
    # checkpoints
    checkpoint:Optional[str] = None
    load_optim:bool = False # l
    load_sched:bool = False

    # Optimization
    learning_rate: float = 5e-6
    grad_accum_steps: int = 1
    mixed_precision: Optional[str] = "no" #['no', 'fp8', 'fp16', 'bf16']
    gradient_checkpointing: bool = True
    total_optimization_steps: int = 100000 # used for linear LR schedule
    warmup_steps: int = 64
    save_step: Optional[int] = 10
    
    action_head_learning_rate: float = 5e-4
    # Weight decay on the ADAPTERS group only (None = torch AdamW's default 0.01, which
    # every run to date has silently carried on every group). Under merge-based init
    # (vlm.merge_adapter_dir) the trainable LoRA is a zero-init delta on the SFT policy,
    # so this decay becomes a standing L2 tether TOWARD the SFT policy -- the second
    # divergence-curbing knob next to rl_config.ref_kl_coeff. Under the legacy
    # parametrization (SFT weights inside the LoRA) it instead decays toward the RAW
    # base -- leave it None there. Not applied to the head group: the head is pretrained,
    # and decaying it pulls toward the zero function, not toward init.
    weight_decay: Optional[float] = None
    # Global grad-norm clip. 1.0 is the discrete-standard value; the chain head's
    # raw norms sit 10-200 (density gradient ~ 1/sigma_step), where a 1.0 clip
    # renormalizes EVERY step and feeds scale jitter into Adam's moments. With
    # sde_position_weight=sigma the raw norm lands ~13, so ~20 clips outliers only.
    max_grad_norm: float = 1.0
    # Freeze everything except the action/value heads (requires_grad False after peft, so
    # the "adapters" optimizer group collects empty and `learning_rate` is inert). The
    # phase-1 latent-RL actor scope (docs/LATENT_RL.md: readout + split, ~5.2M params) and
    # the minimal-drift-surface answer to the measured h-drift collapse channel. Default
    # False: no existing run changes.
    freeze_backbone: bool = False

    # PEFT: Pass the actual configuration object here (e.g., LoraConfig)
    # Typed as Any to avoid crashing if peft isn't installed on the driver
    peft_config: Optional[Any] = field(default_factory=HydraLoraConfig) 

    rl_config:Optional[RLAlgoConfig] = field(default_factory=RLAlgoConfig) # RL Algorithm 
    sft_config:Optional[SFTConfig] = None

# --- Rollouts (both for Eval and RL) ---
@dataclass
class RolloutConfig:
    max_steps: int = 350
    temperature: float = 1.0
    action_space_str: str = "[stop, forward, left, right]"
    system_prompt: str = "${read_text:src/longnav/conf/prompts/objectnav_prompt.txt}"
    action_space: List[str] = field(default_factory=lambda: ["stop", "forward", "left", "right"])
    # Templates are lists of dicts (JSON-like)
    convo_start_template: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"role": "user", "content": [{"type": "text", "text": "${rollout.system_prompt}"}]},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]}
    ])
    
    convo_turn_template: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"role": "assistant", "content": [{"type": "text", "text": "**$action**"}]},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]}
    ])
    stop_prob_threshold: Optional[float] = None
    # Deterministic-rollout mode: act on the env-provided state_dict['info']['oracle_action']
    # instead of sampling from the policy's own output distribution. The policy still runs a
    # real forward pass; only which action is taken (and fed back into the next turn's prompt)
    # is overridden. Used by the forward-pass test tier for reproducible rollouts -- see
    # tests/forward/_bootstrap.py.
    use_oracle_action: bool = False
    # Opportunistic policy/env overlap under RTC (docs/RTC_RL.md section 5): fire the env's
    # begin_interval() -- execute the committed ticks -- without awaiting, before the VLM
    # forward, so the sim moves while the model thinks. Wall-clock only; Ray actor task
    # ordering makes trajectories identical either way. OFF by default, and only valid with
    # an env that exposes begin_interval (the continuous ObjectNav actor with rtc enabled) --
    # enabling it against any other env fails loudly.
    rtc_overlap: bool = False
    # -- blind-episode rejection (worker-side criterion; the COLLECTOR decides whether
    # to honour it, per call: training does, eval does not) --------------------------
    # A step whose geodesic is non-finite ("blind") carries no progress signal: the
    # reward is slack-only and the finite-hold settles the accumulated delta on the
    # recovery step. Measured over 74,692 training episodes: 1.10% of steps, 4.59% of
    # episodes, and 82.8% of affected episodes never recover -- concentrated in a few
    # scenes (43% of one scene's episodes). With this on, an affected episode is
    # DISCARDED and the collector runs a replacement, so no blind row ever reaches
    # collation. OFF by default: a run that does not ask for it behaves exactly as
    # before.
    reject_blind_episodes: bool = False
    # The carve-out, ON by default. A physically-terminated episode (the fall detector
    # fired) goes blind BECAUSE it is falling out of the world: measured, 99.1% of
    # escapes are blind, with a median 2-step blind tail against 25 for other blind
    # episodes. Rejecting them would delete the entire escape population and with it
    # every `escape_penalty` the shaping applies -- silently, while the knob still sat
    # in the YAML. Keep them: they are correctly labelled and correctly penalised, and
    # their blindness is the symptom of a real terminal event, not a measurement gap.
    reject_blind_keep_physical_terminal: bool = True


# --- Experiment housekeeping ---
@dataclass
class RunConfig:
    run_name: str = "debug_run"
    logger: Optional[Any] = None
    shard_size: int = 6
    # Episode source for sharding. EMPTY (the default since 2026-08-24) means the
    # trivial shard: every worker loads the full split itself -- the discrete env's
    # long-standing behaviour and what continuous training did in practice all along
    # (see objectnav_continuous.is_exhausted for the history). The old default,
    # "sample400_a", predates the continuous env's broken bootstrap contract: once that
    # contract was restored, the old default would have silently fed a 400-episode EVAL
    # subset into any training run that did not override it. Name a subset explicitly
    # for pinned evals; never rely on this default for one.
    subset_label: str = ""
    episode_json: str = ""
    output_dir: str = "./dump/results"
    jobtype: str = "eval"
    # --- interleaved train/eval (scripts/train_eval_rl.py only; train_rl.py ignores) ---
    # Run one eval pass every `eval_every` training cycles, over a FIXED, seeded eval set
    # of `eval_set_size` episodes (0 = n_rollout) drawn once from the training pool and
    # partitioned across sims. Fixed set = consecutive points are PAIRED on identical
    # episodes, which is what buys resolution -- a fresh random sample of the same size
    # would bury any real movement under episode variance. `eval_ode` runs the chain head
    # as the pure ODE (the deploy arbiter); eval rollouts never touch the training buffer.
    eval_every: int = 4
    eval_set_size: int = 0
    eval_seed: int = 0
    #: Path to a file of eval uids (comma or newline separated). When set, the eval
    #: partition is these episodes verbatim instead of a seeded draw from the pool --
    #: which is what lets an eval set stay FIXED across runs whose pools differ, and what
    #: lets it be held out of training via the env's `train_uids`.
    eval_uids_file: Optional[str] = None
    eval_ode: bool = True

# --- ROOT CONFIGs ---
# `sim` and `vlm.policy_head` are Hydra ConfigStore groups (see conf/env_configs.py,
# conf/vlm_configs.py), not plain fields. To pick a non-default member:
#   - from the CLI: bare `sim=dummy_discrete` / `policy_head@vlm.policy_head=gaussian_head`
#     (no `+`, since they're already in the defaults list below).
#   - from a yaml (e.g. an experiment/*.yaml under `# @package _global_`): plain
#     `sim: dummy_discrete` does NOT work -- that just assigns the literal string
#     "dummy_discrete" to the field and fails validation once merged against the
#     already-typed group default. Use Hydra's defaults-list override syntax instead:
#       defaults:
#         - override /sim: dummy_discrete
@dataclass
class InferenceConfig:
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    sim: Any = MISSING
    task: RunConfig = field(default_factory=RunConfig)
    defaults: List[Any] = field(default_factory=lambda: ["_self_", {"sim": "habitat"}, {"policy_head@vlm.policy_head": "lm_head"}])

@dataclass
class RLConfig(InferenceConfig):
    training: VLMTrainingConfig = field(default_factory=VLMTrainingConfig)
