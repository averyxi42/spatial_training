"""
A FLOW-MATCHING action head: the whole chunk is generated jointly by integrating a learned
velocity field, instead of being decoded tick by tick from a codebook. This is the baseline
`ar_action_head_v2` is meant to be compared against, and it is written to be a NON-INVASIVE
addition -- nothing in `ar_action_head*.py`, `vector_sft.py`, `turn_vectors.py`,
`vector_rollout.py` or `bin_codec.py` is modified, only imported.

Written against `dump/flow_matching_head/DESIGN.md`, which is the spec; the reference
implementations are `huggingface/lerobot`'s `policies/common/flow_matching.py` and
`policies/pi0/modeling_pi0.py` (openpi-derived). Several conventions below are deliberately
NOT the ones most flow-matching papers use -- see "conventions" -- and the design document
justifies each.

UNRELATED TO `flow_head.py`. That module is a *conditional normalizing flow* trained by
maximum likelihood; this one regresses a velocity field and never evaluates a density.
Different objective, different failure modes, no shared code.

--------------------------------------------------------------------------------------
Why this head exists (the measurement that motivates it)
--------------------------------------------------------------------------------------
On `run_v3_ctx8_pose/checkpoint-2200` the AR head scores R^2 0.9775 teacher-forced and
0.4662 free-running against the same targets. That 0.51 gap is exposure bias: the decoder
conditions on its own drifting samples. This head has NO prefix, NO teacher forcing and
exactly ONE generation mode, so the gap cannot exist -- it is structurally eliminated rather
than mitigated. A linear ridge probe on the same context reaches 0.6303, i.e. the AR head
free-running is currently worse than linear regression on its own input; any head that
closes that is worth knowing about.

--------------------------------------------------------------------------------------
Conventions that are easy to get silently wrong
--------------------------------------------------------------------------------------
1. THE TIME AXIS IS REVERSED relative to most of the literature. openpi's convention:

       x_t = t * noise + (1 - t) * actions        t = 1 is NOISE, t = 0 is DATA
       u_t = noise - actions                      the target velocity, data -> noise
       loss = mse(v_theta(x_t, t, c), u_t)
       inference: Euler from t = 1 down to t = 0, dt = -1/num_steps, x <- x + dt * v

   Adopted verbatim so the numbers are comparable with pi0/pi05/smolVLA and so anyone
   reading those files alongside this one is not misled. `flow_interpolate` and
   `euler_integrate` are the only two places the convention is spelled out, and
   `tests/test_flow_matching_head.py` pins both ends of it plus a full round trip.

2. TIME IS BETA-DISTRIBUTED, NOT UNIFORM. `Beta(1.5, 1.0) * 0.999 + 0.001`. Beta(1.5, 1)
   has density proportional to sqrt(x), so it concentrates near t = 1 -- the high-noise end,
   where the field is hardest and where the first Euler steps live. Substituting uniform
   would be an unjustified deviation from the baseline being replicated.

3. TIME IS FUSED BY CONCATENATION, not addition and not adaLN: a sinusoidal embedding of t
   is concatenated to the projected action and pushed through a 2-layer SiLU MLP, exactly
   pi0's `action_time_mlp_in -> SiLU -> action_time_mlp_out`. No adaLN by default -- pi0
   uses none (`adarms_cond is None`); at d_model=128 x 4 layers, adaLN-Zero modulation
   parameters would be comparable in size to the blocks themselves. It is the first upgrade
   to try if time conditioning turns out to be the weakness, not the starting point.

4. THE ATTENTION IS BLOCKWISE-CAUSAL, copying pi0's `att_masks` semantics: the context block
   is bidirectional among itself, the action block is bidirectional among itself AND attends
   to the context, and the context does NOT attend to the actions. `is_causal=False` is
   MANDATORY for this mask -- `is_causal=True` is only a HINT that permits the SDPA fast
   path to substitute plain causal masking, which would silently discard the mask entirely.
   `ar_action_head._attn_mask` documents the same footgun; there is a test for it here.

5. A LEARNED `tick_embed` IS ADDED. pi0's action tokens get their position from the LM's
   RoPE; this decoder has no positional encoding of any kind, so without `tick_embed` every
   tick would be exchangeable and the chunk would have no temporal shape. The AR decoder
   already carries the same embedding for the same reason.

--------------------------------------------------------------------------------------
Target space: per-tick differentials, then `compose_chunk`
--------------------------------------------------------------------------------------
Same as both existing heads, for a measured reason rather than for symmetry. Anchor-relative
targets were tested on the VQ path and gave 3x the pose error at every tick. The
quantisation half of that argument does not transfer to a continuous head, but the
representational half does: in anchor-relative space "no motion during tick t" is the
RELATION `p_t == p_{t-1}`, a property of two outputs that the model must coordinate, whereas
as a differential it is a property of one output. The measured 44-61x degradation in
near-zero fidelity is evidence that relation is genuinely hard to hold.

The cost is that per-tick errors INTEGRATE along the chunk. Unlike the AR head they do not
COMPOUND through conditioning (generation is joint), so the growth should look like the
codebook's ~4x quantisation profile rather than the AR head's 14.7x free-running profile.
That difference is a measurable prediction of this design.

--------------------------------------------------------------------------------------
ACTION SCALING -- a gap in the design document, resolved here
--------------------------------------------------------------------------------------
The design document does not mention normalising the action space, and taken literally the
recipe does not work: the per-tick differentials have measured stds of ~0.03 m and ~0.05 rad
on v2_25hz, while the noise is standard normal. With `x_t = t*noise + (1-t)*actions` the
data would be a ~3% perturbation of the noise, `u_t = noise - actions` would be dominated by
`noise`, and the network would have to resolve the signal out of the third decimal place of
its own output.

Every openpi-derived policy normalises actions from dataset statistics before flow matching
(pi0 via `max_action_dim` + dataset stats); this is that step, done with the SAME constants
the v2 AR decoder calibrated for its state channel (`ActionDecoderV2.SCALES[2:]`, i.e. 0.03
for xy and 0.05 for theta), so the two heads agree about what "unit scale" means. `xy` gets
ONE shared constant on purpose: a per-axis scale would be anisotropic and would distort the
geometry the decoder reasons over.

The scales live in a BUFFER, so they are saved with the checkpoint and cannot drift between
training and rollout. Pass `action_scales=(1, 1, 1)` to disable, which reproduces the
document's literal recipe (and is expected to train badly -- that is the point of the note).

--------------------------------------------------------------------------------------
The one novel feature: K head forwards per backbone forward
--------------------------------------------------------------------------------------
The flow objective is a Monte Carlo estimate over `(t, noise)`; pi0 draws ONE pair per
example per optimiser step. Here the cost asymmetry is far more extreme than in a typical
VLA: at `--max-turns 400` one backbone forward is a whole episode (~53,000 dense tokens
through 2B parameters) while the head is ~1M parameters over 18 positions. So:

    context (N, C*d) -> repeat_interleave(K) -> (N*K, C*d)
    K stratified times and K noises per context, ONE head forward, loss averaged over K

The benefit is in two places and the second matters more: a lower-variance gradient for the
head's own parameters, and a lower-variance gradient flowing back into the CONTEXT VECTOR,
hence into the readout MLP and the LoRA adapters -- the expensive things to train.

TEMPER THE CLAIM, honestly. One backbone forward already yields up to 400 context vectors,
so the objective already sees ~400 `(t, noise)` draws per step, not one. What restores the
argument is CORRELATION: those 400 turns come from one episode in one scene, so the
effective sample size is far below 400. K is worth having because it is nearly free, not
because it fixes the gradient.

MEMORY CAVEAT, from the same fact. At N=400 and K=16 the expanded head batch is 6,400
sequences of 18 positions; attention is trivial at 18^2 but the activations are not free
(~59 MB per stored tensor, a few GB across 4 layers with gradients). The default is K=8;
chunk the expansion or checkpoint the head if it bites.

STRATIFIED TIME DRAWS, not K i.i.d. Beta samples: one uniform `xi` per example, then
`u_k = (k + xi)/K` mapped through the Beta inverse CDF. Zero extra cost, strictly lower
variance for the t-marginal, and it GUARANTEES every noise level is represented in every
step rather than in expectation. Beta(alpha, 1) has CDF `x^alpha`, so the inverse is
`u^(1/alpha)` in closed form -- which is why `time_beta` must be 1.0 to stratify. Antithetic
noise pairs are a further free variance reduction, behind `antithetic_noise`.

One caveat to document rather than fix: the K samples share a context, so they are not
independent draws from the joint. That is intentional -- we are averaging the CONDITIONAL
loss at fixed `c`, which is exactly the inner expectation the objective is defined over.

--------------------------------------------------------------------------------------
NO STOP / DEADBAND / SOFT-THRESHOLD HEURISTICS. Read the metrics instead.
--------------------------------------------------------------------------------------
The historical creeping failure was a REGRESSION pathology: forward motion is bimodal (stop
or drive), a head trained to predict the conditional mean lands between the modes, and the
robot drifts. The requirement that falls out of it is FAITHFUL MODELLING OF THE WHOLE
CONDITIONAL DISTRIBUTION, not possession of an exact zero atom -- a model that samples from
a faithful `p(a|obs)` puts ~49% of its mass in a tight ball around zero and the robot stops.
Whether the emitted number is bitwise zero or 1e-5 is behaviourally nothing.

So there is no soft-threshold output, no learned deadband and no execution-side threshold
anywhere in this file, and none should be added: any of them would substitute a heuristic
for the property being tested and make the result uninformative.

What DOES need care is READING the metrics. `stop_pred_*` thresholds at `EXACT = 1e-4`,
which a VQ head clears trivially (a centroid sits on zero) and a continuous head may not,
even when behaviourally stopped. Therefore:

  * `stop_pred_*` is NOT comparable between a discrete and a continuous head at face value.
  * the honest comparison is the combined near-zero mass, which both heads can express --
    the trainer logs `near_zero_pred_*` (= stop + creep) against `near_zero_gt_*` for exactly
    this reason. Compare those, not `stop_pred_*`.
  * the behavioural read is whether the robot actually stops, which only the closed-loop
    ObjectNav harness answers.

--------------------------------------------------------------------------------------
THERE IS NO TEACHER-FORCED / FREE-RUNNING SPLIT. Do not compare across heads naively.
--------------------------------------------------------------------------------------
The AR head reports `acc_*` / `rmse_*` teacher-forced and a separate `free_*` family from
sequential decode. This head has ONE generation mode, so that split does not exist and NO
`free_*` metric is emitted -- deliberately, so that nobody diffs a `free_*` key against a
non-`free_*` one. Every number this head logs comes from actually integrating the ODE.

Concretely, when comparing against `ar_action_head_v2`:

    this head's `rmse_dx`        vs  the AR head's `free_rmse_*`-style GENERATION numbers,
                                     NOT its teacher-forced `rmse_dx`
    this head's `pose_rmse_dx`   vs  the AR head's `free_rmse_dx` (both are COMPOSED pose
                                     error against the true chunk, the quantity the
                                     controller actually tracks)
    this head's `rotation_flip`  vs  the AR head's `free_rotation_flip`

`rmse_*`/`mae_*` here are PER-TICK DIFFERENTIAL errors (the same statistic the AR head's
teacher-forced table reports, but generated), and `pose_rmse_*`/`pose_mae_*` are the
composed-pose errors. Both are logged because the two heads' existing tables each cover only
one of them.

--------------------------------------------------------------------------------------
Conditioning: a different division of labour, not a degraded pi0
--------------------------------------------------------------------------------------
pi0's action expert cross-attends the full prefix KV -- every image patch and language
token -- because pi0 assumes the backbone does perception and the expert does the deciding.
This project's premise is the opposite: the backbone output is already ACTION SHAPED, and
the head only translates a latent decision into physics. The evidence is consistent with it:
a LINEAR ridge probe on the context reaches R^2 0.6303 on the whole chunk; widening the
context pathway 8x bought ~10% on loss and nothing on accuracy; the readout MLP discards
0.0007 R^2 relative to the pooled 2048-d backbone state.

Which relocates the null hypothesis: the AR head free-running (0.4662) sitting BELOW a
linear probe (0.6303) on its own input indicts the HEAD, and argues for a more expressive
translator -- this baseline -- rather than for more conditioning.

The genuine open question is whether the trunk supplies the GEOMETRY at the precision the
physics needs, as distinct from supplying the decision ("go left around the chair" does not
say how far the chair is). The diagnostic is direction accuracy holding while `rmse_*` stays
high; the follow-up would be a few extra geometry-carrying context tokens, NOT pi0-style
cross-attention, which would also import pi0's premise about where decisions are made.

STATE TOKEN. pi0 prepends a proprioceptive state token; we have no proprioception. The slot
exists (`use_incoming_motion`), defaults OFF, and is wired to the same incoming-motion
quantity `ActionDecoderV2` scaffolds -- including its VALIDITY BIT, without which an all-zero
vector conflates "genuinely stationary" with "unknown", which imply opposite continuations.
It stays off so that both heads gain or lack it together and the comparison is not broken by
a one-sided change.

--------------------------------------------------------------------------------------
MODALITY EMBEDDING (pose injection), ported from the other two heads
--------------------------------------------------------------------------------------
`forward` overrides the base wholesale, so the injection wiring exists here only because it
was mirrored: `ModalityBatch.pop_from` before the backbone call, `pending` around ONLY that
call, and `zero_touch` added to the loss so every encoder parameter is reached even in a
window with no markers (`ddp_find_unused_parameters=False` hangs otherwise). Inert -- no
extra keys, no extra parameters, no change to the checkpoint -- unless `ModelConfig.
modality_specs` declares something; `dump/flow_pose/flow_inertness_check.py` pins that
against the pre-port commit and `tests/test_flow_pose_injection.py` is its in-tree
companion.

There is NO stop head on this head and none should be added; see the NO STOP / DEADBAND
section above.

`init_modality_from` warm-starts the ENCODERS ALONE from another run's checkpoint (the
regression head's, in practice). That is the only module whose meaning survives a change of
head. What it costs depends on the spec's `gain_init` -- see the method's docstring, which
is worth reading before attributing a lost step-0 guarantee to the warm start.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from longnav.utils.ar_action_head import (  # constants shared with the AR head, by import
    CREEP,
    DEFAULT_DT_NATIVE,
    DIM_NAMES,
    EXACT,
    FLIP_DEADBAND_RADPS,
)
from longnav.utils.ar_action_head_v2 import ActionDecoderV2
from longnav.utils.bin_codec import compose_chunk, decompose_chunk
from longnav.utils.latent_intent import (  # the stochastic-intent delta; see docs/LATENT_RL.md
    LatentConfig, LatentSplit, PosteriorEncoder, kl_shared_sigma, latent_diagnostics,
)
from longnav.utils.modality_embed import ModalityBatch
from longnav.utils.turn_vectors import extract_turn_vectors
from longnav.utils.vector_sft import (
    ADAPTER_SUBDIR,
    HEAD_CONFIG_FILE,
    HEAD_WEIGHTS_FILE,
    LossConfig,
    LoraSpec,
    ModelConfig,
    TurnVectorRegressor,
    TurnVectorSFTTrainer,
    migrate_model_config,
)

HEAD_VERSION = 1

#: openpi's time-sampling law, `Beta(alpha, beta) * scale + offset`. `beta == 1.0` is not
#: incidental: it is what makes the inverse CDF closed form (`u ** (1/alpha)`) and hence
#: what makes the stratified draw free. See `beta_icdf`.
TIME_ALPHA, TIME_BETA = 1.5, 1.0
TIME_SCALE, TIME_OFFSET = 0.999, 0.001
#: pi0's sinusoidal time-embedding periods (`configuration_pi0.min_period/max_period`).
MIN_PERIOD, MAX_PERIOD = 4e-3, 4.0
#: Deploy-time default, logged in `describe()`.
NUM_INFERENCE_STEPS = 10
#: (xy, theta) differential scales, taken from `ActionDecoderV2.SCALES[2:]` so the two heads
#: agree on what unit scale means. See the module docstring's ACTION SCALING section.
ACTION_SCALES: Tuple[float, float, float] = (
    ActionDecoderV2.SCALES[2], ActionDecoderV2.SCALES[2], ActionDecoderV2.SCALES[3],
)


# ======================================================================================
# The flow-matching primitives, isolated so the convention is testable on its own
# ======================================================================================
def sinusoidal_time_embedding(
    time: torch.Tensor, dim: int, min_period: float = MIN_PERIOD,
    max_period: float = MAX_PERIOD,
) -> torch.Tensor:
    """Sine-cosine embedding of a SCALAR position, openpi's `create_sinusoidal_pos_embedding`.

    `time`: (N,). Returns (N, dim) = `[sin(2*pi*t/p), cos(2*pi*t/p)]` over `dim/2` periods
    geometrically spaced between `min_period` and `max_period`.

    Note this is NOT the transformer's usual `10000 ** (2i/d)` schedule: the periods are
    given in the units of `t` itself (which lives in [0, 1]), so `min_period=4e-3` resolves
    time to well under one Euler step at `num_steps=10` while `max_period=4.0` gives a
    monotone, non-wrapping coarse channel across the whole interval. Computed in float64 and
    cast down, exactly as the reference does, because the smallest period is 4e-3 and
    float32 rounding of `2*pi*t/p` is visible there.
    """
    if dim % 2 != 0:
        raise ValueError(f"time embedding dim must be even, got {dim}")
    if time.dim() != 1:
        raise ValueError(f"time must be shape (N,), got {tuple(time.shape)}")
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64, device=time.device)
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * math.pi                       # (dim/2,)
    arg = scaling[None, :] * time.double()[:, None]            # (N, dim/2)
    return torch.cat([torch.sin(arg), torch.cos(arg)], dim=1)


def flow_interpolate(
    actions: torch.Tensor, noise: torch.Tensor, time: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The forward process and its target velocity. `(x_t, u_t)`.

        x_t = t * noise + (1 - t) * actions
        u_t = noise - actions

    **t = 1 IS NOISE AND t = 0 IS DATA** -- reversed from most flow-matching papers, and
    matching openpi / pi0 / pi05 / smolVLA. `u_t` therefore points from data TOWARDS noise
    and is constant in `t` along a path, which is why integrating it with `dt = -1/num_steps`
    (see `euler_integrate`) walks back down to the data.

    `actions`, `noise`: (N, T, 3). `time`: (N,). Returns two (N, T, 3) tensors.
    """
    if time.dim() != 1 or time.shape[0] != actions.shape[0]:
        raise ValueError(
            f"time must be (N,) with N == actions.shape[0]; got {tuple(time.shape)} "
            f"against {tuple(actions.shape)}"
        )
    t = time.to(actions.dtype)[:, None, None]
    return t * noise + (1.0 - t) * actions, noise - actions


def euler_integrate(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    noise: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """Forward-Euler integration from t = 1 (noise) down to t = 0 (actions).

    openpi's loop verbatim, minus the real-time-chunking hook this project does not use:
    `dt = -1/num_steps`, `time = 1 + step*dt`, `x <- x + dt * v(x, time)`. The NEGATIVE `dt`
    is the whole reversed-axis convention in one character; with `u_t = noise - actions` the
    exact field carries `noise` to `actions` in a single step of size 1, which is what
    `tests/test_flow_matching_head.py` checks end to end.

    `velocity_fn(x_t, time)` receives `x_t` of `noise`'s shape and `time` as a float32
    tensor of shape (N,), and must return a velocity of `x_t`'s shape and dtype.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time = torch.full((noise.shape[0],), 1.0 + step * dt,
                          dtype=torch.float32, device=noise.device)
        x_t = x_t + dt * velocity_fn(x_t, time)
    return x_t


def beta_icdf(u: torch.Tensor, alpha: float = TIME_ALPHA,
              beta: float = TIME_BETA) -> torch.Tensor:
    """Inverse CDF of `Beta(alpha, beta)`, closed form and CLOSED FORM ONLY.

    `Beta(alpha, 1)` has density `alpha * x^(alpha-1)` and CDF `x^alpha`, so the inverse is
    `u^(1/alpha)`. That is the whole reason the stratified time draw is free: it needs an
    inverse CDF, torch has no regularised-incomplete-beta inverse, and adding a bisection
    solver to buy a general `beta` would cost more than the variance it saves.

    Raises for `beta != 1`, rather than silently falling back to i.i.d. sampling: the design
    fixes `Beta(1.5, 1.0)`, so a different `beta` means someone changed the law on purpose
    and should be told that stratification is unavailable there.
    """
    if abs(beta - 1.0) > 1e-9:
        raise NotImplementedError(
            f"beta_icdf is closed form only for beta == 1 (got {beta}); Beta(alpha, 1) has "
            "CDF x^alpha. Use stratified_time=False to fall back to i.i.d. Beta draws."
        )
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    return u.clamp(0.0, 1.0).pow(1.0 / alpha)


def sample_time(
    n_examples: int, k_samples: int, *, alpha: float = TIME_ALPHA, beta: float = TIME_BETA,
    scale: float = TIME_SCALE, offset: float = TIME_OFFSET, stratified: bool = True,
    device=None, generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """`(n_examples * k_samples,)` timesteps, ordered to match `repeat_interleave(K)`.

    STRATIFIED (the default): one uniform `xi` per EXAMPLE, then `u_k = (k + xi)/K` mapped
    through `beta_icdf`. Every example therefore gets exactly one time in each of the K
    equal-probability strata, so every noise level is represented at every optimiser step
    rather than in expectation. A single shared `xi` (systematic sampling) is what the design
    document specifies; it makes the K draws perfectly rank-correlated, which is the point --
    each `u_k` is still marginally uniform on its stratum, so the estimator stays unbiased.

    I.I.D.: K independent Beta draws, i.e. what pi0 does K times. Kept for the ablation that
    measures what stratification actually buys.

    The returned layout is `[ex0_k0, ex0_k1, ..., ex0_kK-1, ex1_k0, ...]`, which is exactly
    the order `context.repeat_interleave(K, dim=0)` produces. Getting this wrong would pair
    every time with the wrong context and is invisible in the loss curve.
    """
    if k_samples < 1:
        raise ValueError(f"k_samples must be >= 1, got {k_samples}")
    if stratified:
        xi = torch.rand(n_examples, 1, device=device, generator=generator)
        k = torch.arange(k_samples, device=device, dtype=xi.dtype)[None, :]
        base = beta_icdf((k + xi) / k_samples, alpha, beta)
    elif abs(beta - 1.0) <= 1e-9:
        base = beta_icdf(
            torch.rand(n_examples, k_samples, device=device, generator=generator),
            alpha, beta,
        )
    else:  # general Beta; torch's sampler does not accept a generator, hence the branch
        dist = torch.distributions.Beta(torch.tensor(float(alpha)), torch.tensor(float(beta)))
        base = dist.sample((n_examples, k_samples)).to(device)
    return (base * scale + offset).reshape(-1).to(dtype)


def sample_noise(
    n_examples: int, k_samples: int, n_ticks: int, n_dims: int = 3, *,
    antithetic: bool = False, device=None, generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """`(n_examples * k_samples, n_ticks, n_dims)` standard-normal noise, same layout.

    `antithetic` pairs stratum `k` with stratum `k + K/2` as `(eps, -eps)`. Free variance
    reduction for any velocity field whose error is odd in the noise; requires an even K.
    """
    if antithetic:
        if k_samples % 2 != 0:
            raise ValueError(f"antithetic noise needs an even k_samples, got {k_samples}")
        half = torch.randn(n_examples, k_samples // 2, n_ticks, n_dims,
                           device=device, generator=generator, dtype=dtype)
        eps = torch.cat([half, -half], dim=1)
    else:
        eps = torch.randn(n_examples, k_samples, n_ticks, n_dims,
                          device=device, generator=generator, dtype=dtype)
    return eps.reshape(n_examples * k_samples, n_ticks, n_dims)


# ======================================================================================
# Config
# ======================================================================================
@dataclass
class FlowMatchingConfig:
    """Everything about the OBJECTIVE and the SAMPLER, as distinct from the network's shape
    (which lives in `FlowActionDecoder.to_config()`). Round-trips through the checkpoint's
    `turn_vector_head_config.json` so an eval run cannot silently use a different time law
    or a different number of Euler steps than the run that trained the weights."""

    #: head forwards per backbone forward. See the module docstring's memory caveat.
    k_samples: int = 8
    #: Euler steps at deploy time. A knob, not a trained quantity.
    num_inference_steps: int = NUM_INFERENCE_STEPS
    #: Euler steps used for the metrics logged during training/eval. Kept separate so the
    #: logged numbers can be pinned while `num_inference_steps` is swept at eval.
    metric_inference_steps: int = NUM_INFERENCE_STEPS
    time_alpha: float = TIME_ALPHA
    time_beta: float = TIME_BETA
    time_scale: float = TIME_SCALE
    time_offset: float = TIME_OFFSET
    stratified_time: bool = True
    antithetic_noise: bool = False
    #: per-dimension divisor applied to the differentials before flow matching; see the
    #: module docstring's ACTION SCALING section. `(1, 1, 1)` disables it.
    action_scales: Tuple[float, float, float] = ACTION_SCALES

    def __post_init__(self):
        self.action_scales = tuple(float(s) for s in self.action_scales)
        if len(self.action_scales) != 3 or any(s <= 0 for s in self.action_scales):
            raise ValueError(f"action_scales must be 3 positive floats, got {self.action_scales}")
        if self.k_samples < 1:
            raise ValueError(f"k_samples must be >= 1, got {self.k_samples}")
        if self.stratified_time and abs(self.time_beta - 1.0) > 1e-9:
            raise ValueError(
                "stratified_time requires time_beta == 1.0 (the Beta(alpha, 1) inverse CDF "
                "is what makes stratification free); pass stratified_time=False to use "
                "i.i.d. draws with a general beta"
            )


# ======================================================================================
# The velocity field
# ======================================================================================
class FlowActionDecoder(nn.Module):
    """`v_theta(x_t, t, context) -> (N, T, 3)`, a tiny transformer over

        [ ctx_0 .. ctx_{C-1} | (state) | a_0 .. a_{T-1} ]

    shaped to mirror `ActionDecoderV2` so capacity is comparable: `d_model=128, n_layers=4,
    n_heads=4, dim_ff=512`, same `nn.TransformerEncoderLayer` (pre-norm, GELU). It lands
    ~263k parameters SMALLER than the AR decoder because there is no 1024-way codebook
    embedding and no 1024-way output projection -- reported rather than padded away with a
    wider `dim_ff`, since the shortfall is exactly the discrete machinery this head does not
    have.

    CONTEXT PATHWAY. As in v2 there is NO context projection: the readout MLP emits
    `n_context_tokens * d_model` and this module only reshapes. The constructor enforces the
    width agreement rather than warning about it, because a projection sitting immediately
    after the readout MLP's final Linear with no activation between them collapses to one
    linear map (v1's measured ~1.05M dead parameters).

    ACTION TOKEN. `action_in_proj(x_t[t])` concatenated with `sinusoidal_time_embedding(t)`,
    through `action_time_mlp_in -> SiLU -> action_time_mlp_out` (pi0's fusion, verbatim),
    plus a learned `tick_embed(t)`. The tick embedding is added AFTER the fusion MLP because
    that is where pi0's positional information enters (its LM applies RoPE to the fused
    embedding), and because the MLP is a per-token map that should see the action and the
    time, not a positional code it would have to learn to ignore.

    OUTPUT. `out_proj(ln_f(h))` at the action positions = the velocity, in the SCALED action
    space (`FlowActionCodec` owns the scaling).
    """

    #: `[dx, dy, dtheta, valid]` carried into tick 0 from the previous chunk, matching
    #: `ActionDecoderV2._state`'s incoming-motion slot exactly, validity bit included.
    STATE_DIM = 4

    def __init__(self, context_dim: int, n_ticks: int, d_model: int = 128,
                 n_layers: int = 4, n_heads: int = 4, dim_ff: int = 512,
                 dropout: float = 0.1, n_context_tokens: int = 8, n_dims: int = 3,
                 use_incoming_motion: bool = False, min_period: float = MIN_PERIOD,
                 max_period: float = MAX_PERIOD,
                 state_scales: Optional[Sequence[float]] = None):
        super().__init__()
        want = int(n_context_tokens) * d_model
        if context_dim != want:
            raise ValueError(
                f"this head has no context projection: the readout MLP must emit exactly "
                f"n_context_tokens * d_model ({n_context_tokens} * {d_model} = {want}), "
                f"got context_dim={context_dim}"
            )
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for the time embedding, got {d_model}")
        self.n_ticks, self.d_model, self.n_dims = int(n_ticks), int(d_model), int(n_dims)
        self.n_context_tokens = int(n_context_tokens)
        self.context_dim = int(context_dim)
        self.use_incoming_motion = bool(use_incoming_motion)
        self.min_period, self.max_period = float(min_period), float(max_period)
        # Only used to bring the incoming differential to ~unit scale; same constants as the
        # v2 AR decoder's differential channel, so the slot means the same thing in both.
        self.state_scales = tuple(float(x) for x in
                                  (state_scales or ActionDecoderV2.SCALES[2:]))
        if len(self.state_scales) != 2 or any(x <= 0 for x in self.state_scales):
            raise ValueError(f"state_scales must be 2 positive floats, got {state_scales!r}")

        self.action_in_proj = nn.Linear(self.n_dims, d_model)
        self.action_time_mlp_in = nn.Linear(2 * d_model, d_model)
        self.action_time_mlp_out = nn.Linear(d_model, d_model)
        # The ONLY positional signal in this decoder: there is no RoPE and no sinusoidal
        # position code, so without this every tick would be exchangeable.
        self.tick_embed = nn.Embedding(self.n_ticks, d_model)
        if self.use_incoming_motion:
            self.state_norm = nn.LayerNorm(self.STATE_DIM)
            self.state_proj = nn.Linear(self.STATE_DIM, d_model)
        else:
            self.state_norm = self.state_proj = None
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_ff, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, self.n_dims)
        self._init_kwargs = dict(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads, dim_ff=dim_ff,
            dropout=dropout, n_context_tokens=self.n_context_tokens, n_dims=self.n_dims,
            use_incoming_motion=self.use_incoming_motion, min_period=self.min_period,
            max_period=self.max_period, state_scales=list(self.state_scales),
        )

    def to_config(self) -> Dict:
        return dict(self._init_kwargs)

    @property
    def n_state_tokens(self) -> int:
        return 1 if self.use_incoming_motion else 0

    @property
    def seq_len(self) -> int:
        return self.n_context_tokens + self.n_state_tokens + self.n_ticks

    def _attn_mask(self, device, dtype) -> Tuple[torch.Tensor, bool]:
        """BLOCKWISE-CAUSAL additive mask over `seq_len` positions, and the `is_causal` hint.

        pi0's `att_masks` semantics, reduced to the two blocks this head has:

          * the CONTEXT block is bidirectional among itself and sees nothing else;
          * the ACTION block (the optional state token plus every tick) is bidirectional
            among itself AND attends to the whole context block.

        So the only `-inf` quadrant is context-queries x non-context-keys. The action ticks
        must see each other -- that bidirectionality is the entire reason this head has no
        exposure bias, and a causal mask here would quietly turn it into a worse AR head.

        `is_causal` is returned FALSE, always and by construction. It is only a HINT that
        permits the SDPA fast path to apply plain causal masking, which would DISCARD this
        mask silently; the same footgun is documented in `ar_action_head._attn_mask`, and
        `tests/test_flow_matching_head.py` asserts it here.

        The state token is placed in the ACTION block (rather than in a block of its own as
        pi0 does) so the mask stays the single quadrant the design document specifies. The
        only difference that makes is that the state token can read the ticks; it is off by
        default and carries no target, so nothing leaks.
        """
        C, L = self.n_context_tokens, self.seq_len
        mask = torch.zeros(L, L, dtype=dtype, device=device)
        mask[:C, C:] = float("-inf")
        return mask, False

    def _state_token(self, incoming: Optional[torch.Tensor], n: int,
                     dtype, device) -> torch.Tensor:
        """(N, 1, d_model) from `incoming` = (N, 4) `[dx, dy, dtheta, valid]`.

        Absent means UNKNOWN, encoded as zeros with `valid = 0` -- not as "at rest". Without
        the validity bit those two are the same vector and they imply opposite continuations:
        under a 50%-overlapping receding horizon the base is mid-motion when a chunk is
        predicted, so "unknown" is the common case and "at rest" is the rare one.
        """
        if incoming is None:
            incoming = torch.zeros(n, self.STATE_DIM, dtype=dtype, device=device)
        inc = incoming.to(dtype)
        s_xy, s_th = self.state_scales
        feats = torch.stack([inc[:, 0] / s_xy, inc[:, 1] / s_xy,
                             inc[:, 2] / s_th, inc[:, 3]], dim=-1)
        return self.state_proj(self.state_norm(feats)).unsqueeze(1)

    def forward(self, context: torch.Tensor, x_t: torch.Tensor, time: torch.Tensor,
                incoming: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`context`: (N, context_dim). `x_t`: (N, T, 3) in SCALED action space. `time`:
        (N,) in [0, 1], where 1 is noise. Returns (N, T, 3), the velocity at `x_t`.

        Every one of the N rows is independent; the K-sample expansion is done by the caller
        (`TurnFlowActionRegressor.forward`) precisely so this stays a plain batched forward.
        """
        N, T = x_t.shape[0], self.n_ticks
        if x_t.shape[1:] != (T, self.n_dims):
            raise ValueError(
                f"x_t must be (N, {T}, {self.n_dims}), got {tuple(x_t.shape)}"
            )
        if context.shape[0] != N:
            raise ValueError(
                f"context has {context.shape[0]} rows but x_t has {N}; the K-sample "
                "expansion must be applied to BOTH (see repeat_interleave in the trainer)"
            )
        device, C, d = context.device, self.n_context_tokens, self.d_model
        ctx_tok = context.view(N, C, d)                      # reshape only -- no projection

        act = self.action_in_proj(x_t)                       # (N, T, d)
        t_emb = sinusoidal_time_embedding(
            time, d, self.min_period, self.max_period).to(act.dtype)
        t_emb = t_emb[:, None, :].expand_as(act)             # broadcast over ticks
        fused = self.action_time_mlp_out(
            F.silu(self.action_time_mlp_in(torch.cat([act, t_emb], dim=-1)))
        )
        fused = fused + self.tick_embed(torch.arange(T, device=device))[None]

        parts = [ctx_tok]
        if self.use_incoming_motion:
            parts.append(self._state_token(incoming, N, act.dtype, device))
        parts.append(fused)
        x = torch.cat(parts, dim=1)                          # (N, seq_len, d)

        mask, is_causal = self._attn_mask(device, x.dtype)
        h = self.blocks(x, mask=mask, is_causal=is_causal)
        return self.out_proj(self.ln_f(h[:, -T:, :]))        # (N, T, 3)


# ======================================================================================
# The normalizer-slot adapter: the Euler loop, one `denormalize` call deep
# ======================================================================================
class FlowActionCodec(nn.Module):
    """Occupies the `normalizer` slot of `TurnVectorRegressor`'s contract exactly the way
    `ARActionCodec` does, so `save_pretrained` / `load_head_state` and
    `VectorRolloutPolicy.step()` work with NO change to `vector_sft.py` or
    `vector_rollout.py`. The substitution happens in `denormalize(context) -> chunk`: instead
    of one function call on logits it runs the full Euler loop from t=1 to t=0 and composes
    the resulting differentials.

    It also owns the ACTION SCALING (see the module docstring). Keeping the scales here
    rather than in the network means (a) they are one buffer in the checkpoint that training
    and rollout cannot disagree about, and (b) the network is a pure velocity field over a
    unit-scale space with no physical constants baked into its weights.

    NO AVERAGING OF SAMPLES. `DECODES` deliberately has no `mean`: averaging several ODE
    samples reproduces the conditional mean, which IS the creeping failure this whole line
    of work is about. One draw, or best-of-k under an explicit criterion -- never a mean.
    """

    #: `context` is the passthrough that lets an offline script save the cheap-to-decode
    #: quantity once per real VLM forward and run every decode rule later on CPU, exactly
    #: the role `ARActionCodec`'s `"context"` plays for the AR head.
    DECODES = ("sample", "context")

    def __init__(self, decoder: FlowActionDecoder,
                 num_inference_steps: int = NUM_INFERENCE_STEPS,
                 action_scales: Sequence[float] = ACTION_SCALES,
                 latent: Optional[LatentSplit] = None):
        super().__init__()
        self.decoder = decoder
        #: The stochastic intent variable `c`, or None for v3's deterministic behaviour. It
        #: lives HERE rather than on the regressor because `denormalize(context) -> chunk`
        #: is the rollout entry point, so putting the split behind it means the rollout,
        #: every eval backend and the policy bridge pick it up with NO code change -- the
        #: same reason this class occupies the normalizer slot at all. `latent_mode`
        #: defaults to `mean`, which is bit-for-bit the deterministic path.
        self.latent = latent
        self.latent_mode = "mean"
        #: When set, EVERY decode reuses this base noise instead of drawing a fresh one.
        #: This is the RL-time actuator: with `z_0` pinned and the ODE step count fixed,
        #: `c -> chunk` is deterministic and differentiable, which is what lets a critic
        #: gradient reach the policy. It also makes `latent_mode="sample"` mean what RL
        #: means by it -- vary the INTENT while execution is held fixed. Without pinning,
        #: sample mode varies `c` and the flow noise together (they share `self.generator`),
        #: which measures something the RL policy will never do.
        self.pinned_flow_noise: Optional[torch.Tensor] = None
        scales = torch.tensor([float(s) for s in action_scales], dtype=torch.float64)
        if scales.numel() != decoder.n_dims or bool((scales <= 0).any()):
            raise ValueError(f"action_scales must be {decoder.n_dims} positive floats")
        self.register_buffer("action_scales", scales)
        self.decode = "sample"
        self.num_inference_steps = int(num_inference_steps)
        #: settable seedable generator, matching the AR head's sample-decode pattern, so a
        #: rollout is reproducible. It must live on the same device as the context.
        self.generator: Optional[torch.Generator] = None

    # -- the scaling, in both directions ------------------------------------------------
    def scale(self, diffs: torch.Tensor) -> torch.Tensor:
        """physical differentials -> the unit-ish space the velocity field lives in."""
        return diffs / self.action_scales.to(diffs.device, diffs.dtype)

    def unscale(self, x: torch.Tensor) -> torch.Tensor:
        """the inverse of `scale`."""
        return x * self.action_scales.to(x.device, x.dtype)

    def normalize(self, chunk: torch.Tensor) -> torch.Tensor:
        """(..., T, 3) anchor-relative chunk -> (..., T, 3) SCALED per-tick differentials,
        i.e. the `actions` the flow objective interpolates towards."""
        return self.scale(decompose_chunk(chunk.double()))

    # -- generation ---------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, context: torch.Tensor, num_steps: Optional[int] = None,
                 generator: Optional[torch.Generator] = None,
                 noise: Optional[torch.Tensor] = None,
                 incoming: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(N, context_dim) -> (N, T, 3) PHYSICAL per-tick differentials.

        Draws `x_1 ~ N(0, I)` in the scaled space and integrates the field down to t = 0.
        `generator` (else `self.generator`, else the global RNG) seeds the draw; it must be a
        generator for `context.device`, since that is where the noise is allocated.

        There is exactly ONE generation mode. No teacher forcing exists for this head, so
        this is both the training-time metric path and the deployed path -- which is the
        entire structural claim being tested.
        """
        steps = int(num_steps or self.num_inference_steps)
        gen = generator if generator is not None else self.generator
        dec = self.decoder
        ctx = context.float()
        if noise is None:
            noise = torch.randn(ctx.shape[0], dec.n_ticks, dec.n_dims,
                                device=ctx.device, dtype=ctx.dtype, generator=gen)
        x0 = euler_integrate(
            lambda x_t, t: dec(ctx, x_t, t, incoming=incoming), noise, steps
        )
        return self.unscale(x0.double())

    def denormalize(self, context: torch.Tensor) -> torch.Tensor:
        """(N, context_dim) -> (N, T, 3) anchor-relative chunk. The rollout entry point.

        `context` is `h`, the deterministic readout. With a latent installed it is mapped to
        the intent `c` first -- always from the PRIOR, since there is no ground-truth chunk
        at rollout time and the posterior does not exist outside SFT. `latent_mode="mean"`
        reproduces the deterministic path exactly; `"sample"` is what an RL rollout uses.
        """
        if self.decode not in self.DECODES:
            raise ValueError(f"decode must be one of {self.DECODES}, got {self.decode!r}")
        if self.decode == "context":
            return context
        ctx = context
        noise = None
        if self.latent is not None:
            ctx = self.latent.draw(context.float(), mode=self.latent_mode,
                                   generator=self.generator)["c"]
            if self.pinned_flow_noise is not None:
                noise = self.pinned_flow_noise.to(ctx.device, torch.float32)
                noise = noise.expand(ctx.shape[0], *noise.shape[1:])
        elif self.latent_mode != "mean":
            raise ValueError(
                f"latent_mode={self.latent_mode!r} on a codec with no latent split: there "
                "is no `c` to sample. This checkpoint is deterministic."
            )
        return compose_chunk(self.generate(ctx, noise=noise))

    def pin_flow_noise(self, seed: Optional[int]) -> None:
        """Freeze the ODE's base noise at `seed`, or clear it with `None`.

        Separate from `seed_sampling`, which reseeds the generator that draws a FRESH noise
        per decode. Both are reproducible; only this one makes execution constant across
        decodes, which is the difference between "the policy is stochastic" and "the
        intent is stochastic and the actuator is not".
        """
        if seed is None:
            self.pinned_flow_noise = None
            return
        dev = self.action_scales.device
        gen = torch.Generator(device=dev if dev.type == "cuda" else "cpu")
        gen.manual_seed(int(seed))
        self.pinned_flow_noise = torch.randn(
            1, self.decoder.n_ticks, self.decoder.n_dims,
            device=gen.device, dtype=torch.float32, generator=gen,
        ).to(dev)

    def denormalize_from_latent(self, c: torch.Tensor,
                                noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`c` (already an intent, already in the decoder's basis) -> anchor-relative chunk.

        The seam the spread probe needs and `denormalize` cannot provide: that method maps
        `h -> c` itself, so it can never decode a `c` the caller drew, and it owns the base
        noise, so it cannot hold the noise fixed across draws. Both are exactly what the
        acceptance gate has to vary independently -- vary `c` at fixed noise, then vary
        noise at fixed `c`. This is also the RL-time actuator: pass a pinned `noise` and the
        map is deterministic and differentiable in `c`.
        """
        return compose_chunk(self.generate(c, noise=noise))

    def describe(self) -> str:
        """One line for the run log / `predict_*` tooling: the deploy-time knobs that change
        what the SAME weights produce."""
        # The latent state is printed unconditionally, including when there is no latent.
        # A latent checkpoint silently evaluated in `mean` mode is indistinguishable in its
        # results from a deterministic one, so the run record has to say which it was.
        latent = ("none" if self.latent is None
                  else f"mode={self.latent_mode} pinned_noise="
                       f"{self.pinned_flow_noise is not None}")
        return (f"flow-matching head: decode={self.decode} "
                f"num_inference_steps={self.num_inference_steps} "
                f"action_scales={tuple(self.action_scales.tolist())} "
                f"seeded={self.generator is not None} latent[{latent}]")


# ======================================================================================
# The model
# ======================================================================================
class TurnFlowActionRegressor(TurnVectorRegressor):
    """`TurnVectorRegressor` whose head emits a context vector consumed by a
    `FlowActionDecoder`, with a flow-matching objective instead of Huber-over-reals.

    Three changes relative to the base class, the same three `ar_action_head` makes:

      * `target_shape` is `(context_dim,)`, so the shared readout MLP (`TurnVectorHead`) is
        repurposed to emit a per-turn CONTEXT VECTOR rather than the 30 numbers directly;
      * the `normalizer` slot holds a `FlowActionCodec`, whose `denormalize` runs the Euler
        loop -- so rollout works unchanged;
      * `forward` computes the flow-matching MSE, averaged over K `(t, noise)` draws per
        turn (the K-sample expansion described in the module docstring).

    NO `free_*` METRICS ARE EMITTED. There is one generation mode; see the module
    docstring's section on that, and do not compare these numbers against the AR head's
    teacher-forced table.
    """

    #: mirrors `TurnARActionClassifier.DECODER_CLS`, so a variant can swap the field
    #: without forking `build`/`from_pretrained`.
    DECODER_CLS = FlowActionDecoder

    @classmethod
    def build(
        cls,
        model_cfg: ModelConfig,
        loss_cfg: LossConfig,
        lora: Optional[LoraSpec],
        n_ticks: int,
        processor,
        context_dim: int = 1024,
        decoder_kwargs: Optional[Dict] = None,
        fm_cfg: Optional[FlowMatchingConfig] = None,
        dtype: torch.dtype = torch.bfloat16,
        latent_cfg: Optional[LatentConfig] = None,
    ) -> "TurnFlowActionRegressor":
        loss_cfg = LossConfig(**{**loss_cfg.__dict__, "normalize_targets": False})
        target_shape = (context_dim,)   # sizes the readout MLP's projection only
        model = super().build(model_cfg, loss_cfg, lora, target_shape, processor, dtype)
        fm_cfg = fm_cfg or FlowMatchingConfig()
        decoder = cls.DECODER_CLS(context_dim=context_dim, n_ticks=n_ticks,
                                  **(decoder_kwargs or {}))
        latent = None
        if latent_cfg is not None and latent_cfg.enabled:
            latent = LatentSplit(dim=context_dim, sigma0=latent_cfg.sigma0,
                                 rotation=latent_cfg.rotation)
        model.normalizer = FlowActionCodec(
            decoder, num_inference_steps=fm_cfg.num_inference_steps,
            action_scales=fm_cfg.action_scales, latent=latent,
        )
        model.n_ticks = int(n_ticks)
        model.fm_cfg = fm_cfg
        model.latent_cfg = latent_cfg
        # SFT-only, and deliberately NOT on the codec: the codec is what rollout and every
        # eval backend load, and the posterior must not be reachable from there.
        model.posterior = (
            PosteriorEncoder(dim=context_dim, n_ticks=n_ticks,
                             n_dims=decoder.n_dims, width=latent_cfg.posterior_width)
            if latent is not None else None
        )
        return model

    @property
    def decoder(self) -> FlowActionDecoder:
        return self.normalizer.decoder

    @property
    def codec(self) -> FlowActionCodec:
        """The normalizer-slot codec, i.e. what `.decode` / `.generator` /
        `.num_inference_steps` live on.

        NOTE THE NAMING DIFFERENCE, because it bites when writing an eval backend:
        `TurnBinClassifier.codec` is also the normalizer-slot object, but
        `TurnARActionClassifier.codec` is the VQ CODEBOOK and its generator lives on
        `.normalizer` instead. This head has no lookup table, so `.codec` is the
        normalizer-slot object here -- the `bin_rollout.seed_sampling` pattern
        (`policy.model.codec.generator = ...`) transfers directly, with
        `codec.action_scales.device` standing in for `codec.centroids.device`.
        """
        return self.normalizer

    # -- the objective -----------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_turns: Optional[torch.Tensor] = None,
        num_items_in_batch: Optional[Union[int, torch.Tensor]] = None,
        **backbone_inputs,
    ) -> Dict[str, torch.Tensor]:
        """The flow-matching objective, wrapped in the same modality-injection wiring
        `TurnVectorRegressor.forward` and `TurnARActionClassifier.forward` carry.

        `backbone_inputs` is whatever else the collator produced (`pixel_values`,
        `image_grid_thw`, ...) and is forwarded to the backbone **wholesale** -- which is
        exactly why the `modality_*` keys have to come out first and by name. The backbone
        does not accept them, so anything left behind is a `TypeError` from deep inside the
        model; `pop_from` owns the whole `modality_*` prefix and raises on a key it does not
        recognise rather than letting one through.

        NO STOP HEAD here, unlike the other two heads: this head's own answer to the
        creeping failure is modelling the whole conditional distribution (see the module
        docstring), so a second binary readout would be a heuristic bolted onto the thing
        being tested. `self.head` therefore stays inside `extract_turn_vectors` rather than
        being applied explicitly -- there is no second consumer of the pooled state to keep
        in agreement with it.
        """
        backbone_inputs.pop("labels", None)
        modality = ModalityBatch.pop_from(
            backbone_inputs, known_keys=self.modality_embedder.keys
        )
        # The context wraps only the backbone call: entering the embedding twice under one
        # context trips the consume-once assert instead of silently reusing the values.
        with self.modality_embedder.pending(modality):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
                **backbone_inputs,
            )
        context, spans = extract_turn_vectors(
            outputs,
            input_ids,
            self.head,
            prefix_ids=self.prefix_ids,
            postfix_ids=self.postfix_ids,
            shift_left=self.model_cfg.shift_left,
            strict=True,
        )
        if context.shape[0] != targets.shape[0]:
            tail = input_ids[0, -64:].tolist()
            raise RuntimeError(
                f"found {context.shape[0]} assistant turn span(s) but got "
                f"{targets.shape[0]} target(s). prefix={self.model_cfg.prefix!r} "
                f"postfix={self.model_cfg.postfix!r} "
                f"shift_left={self.model_cfg.shift_left}. Last 64 input_ids: {tail}"
            )
        if self.train_content_len is None and spans:
            self.train_content_len = int(len(spans[0]))

        cfg = self.fm_cfg
        codec, dev = self.normalizer, context.device
        targets = targets.to(dev)
        gt_diffs = decompose_chunk(targets.double())                  # (N, T, 3) physical
        actions = codec.scale(gt_diffs).float()                       # (N, T, 3) unit-ish
        N, T, D = actions.shape
        K = cfg.k_samples

        # ---- the latent intent. `context` is `h`; what the decoder consumes is `c`.
        # Without a latent installed this block is inert and `c is h`, so a deterministic
        # run is byte-identical to before this existed.
        latent, kl, metrics_ctx = codec.latent, None, context.float()
        if latent is None:
            c = context.float()
        else:
            # The posterior sees `h` DETACHED: its parameters are discarded at RL and must
            # not shape the trunk. It also sees `h` rather than the pooled state, so its
            # information set matches the prior's -- see latent_intent's module docstring.
            delta_mu = self.posterior(context.float().detach(), actions)
            pieces = latent.draw(context.float(), mode="sample", delta_mu=delta_mu)
            c = pieces["c"]
            kl = kl_shared_sigma(delta_mu, pieces["log_sigma"])        # (N,) nats
            # Metrics report the DEPLOYED path, which is the prior at its mean.
            metrics_ctx = latent.decode(pieces["mu"])
            with torch.no_grad():
                latent_metrics = latent_diagnostics(pieces, delta_mu, kl, context.float())

        # ---- the K-sample expansion. One head forward on N*K rows; see the module
        # docstring. `repeat_interleave` (not `repeat`) so row order matches `sample_time`'s
        # and `sample_noise`'s layout, and so the gradient of each context vector receives
        # the average of its own K draws.
        #
        # `c` IS DRAWN ONCE PER ROW AND THEN EXPANDED, never drawn K times. The K draws are
        # the flow's own (t, noise) integration samples; giving each of them a different
        # intent would average the reconstruction over intents and change what the gradient
        # reaching `sigma` means. Invisible if wrong.
        ctx_k = c.repeat_interleave(K, dim=0)                         # (N*K, context_dim)
        act_k = actions.repeat_interleave(K, dim=0)                   # (N*K, T, 3)
        time = sample_time(N, K, alpha=cfg.time_alpha, beta=cfg.time_beta,
                           scale=cfg.time_scale, offset=cfg.time_offset,
                           stratified=cfg.stratified_time, device=dev)
        noise = sample_noise(N, K, T, D, antithetic=cfg.antithetic_noise, device=dev)
        x_t, u_t = flow_interpolate(act_k, noise, time)
        v_t = self.decoder(ctx_k, x_t, time)                          # (N*K, T, 3)

        per_draw = F.mse_loss(v_t, u_t, reduction="none").flatten(1).mean(dim=1)
        per_turn = per_draw.view(N, K).mean(dim=1)                    # average over K
        if kl is not None:
            # ELBO in shape only: `per_turn` is a velocity regression, not -log p(A|c), so
            # `beta` is a bare exchange rate with no transferable scale and must be swept.
            # The KL is still in nats, which is why the diagnostics on it mean something.
            per_turn = per_turn + float(self.latent_cfg.beta) * kl
        total = per_turn.sum()
        denom = num_items_in_batch if num_items_in_batch is not None else per_turn.numel()
        denom = torch.as_tensor(denom, dtype=total.dtype, device=total.device).clamp(min=1)
        loss = total / denom

        # An example with no occurrences of a modality leaves its encoder out of the
        # backward graph, which under `ddp_find_unused_parameters=False` hangs or errors.
        # Identically zero -- no gradient and no metric moves -- and exists only so every
        # encoder parameter is always reached.
        touch = self.modality_embedder.zero_touch(loss.device, loss.dtype)
        if touch is not None:
            loss = loss + touch

        with torch.no_grad():
            metrics = self._generation_metrics(metrics_ctx, gt_diffs, targets)
            if kl is not None:
                metrics.update({k: v.detach() for k, v in latent_metrics.items()})
                metrics["fm_loss_sum"] = per_draw.view(N, K).mean(dim=1).sum().detach()
            metrics["loss_sum"] = total.detach()
            metrics["n_turns"] = torch.tensor(N, device=dev)
            metrics["n_tokens"] = torch.tensor(
                outputs["last_hidden_state"].shape[1], device=dev
            )
            metrics["n_dense_tokens"] = torch.tensor(input_ids.shape[1], device=dev)
            metrics["n_steps"] = torch.tensor(1, device=dev)
        return {"loss": loss, **metrics}

    @torch.no_grad()
    def _generation_metrics(self, context, gt_diffs, targets) -> Dict[str, torch.Tensor]:
        """Everything the trainer logs, computed from ONE actual ODE integration.

        Unlike the AR head there is no cheap teacher-forced table to report every step and an
        expensive free-running one to report at eval, so this runs on EVERY step: it costs
        `metric_inference_steps` forwards of a ~1M-parameter network over 18 positions, which
        is nothing beside the backbone forward that produced `context`. It also has to run
        every step because `TurnVectorSFTTrainer._drain_metrics` returns nothing at all
        without `n_rows`, so a metrics-at-eval-only variant would log no loss either.

        `decoder.eval()` for the duration: the training-mode dropout would otherwise perturb
        the integration, and a metric of the deployed path should measure the deployed path.

        What is reported, and against what:

          `rmse_*` / `mae_*`      PER-TICK DIFFERENTIAL error, the same statistic as the AR
                                  head's `rmse_*` -- but GENERATED, not teacher-forced.
          `pose_rmse_*`/`pose_mae_*`  COMPOSED pose error against the true chunk, the
                                  quantity the controller tracks, comparable with the AR
                                  head's `free_rmse_*`.
          `stop_*` / `creep_*`    band mass. `stop_pred_*` is NOT comparable at face value
                                  between a discrete and a continuous head (see the module
                                  docstring); `near_zero_*` is the comparable one.
          `rotation_flip`         same definition and same deadband as the AR head's
                                  `free_rotation_flip` and the closed-loop probes.
        """
        dev = context.device
        was_training = self.decoder.training
        self.decoder.eval()
        try:
            diffs = self.normalizer.generate(
                context, num_steps=self.fm_cfg.metric_inference_steps
            )                                                          # (N, T, 3) physical
        finally:
            self.decoder.train(was_training)
        chunk = compose_chunk(diffs)                                   # (N, T, 3)

        D = len(DIM_NAMES)
        creep = torch.tensor(CREEP, dtype=torch.float64, device=dev)

        def bands(v):
            a = v.abs()
            stop = (a < EXACT).reshape(-1, D).double().sum(0)
            crp = ((a >= EXACT) & (a < creep)).reshape(-1, D).double().sum(0)
            return stop, crp

        stop_p, creep_p = bands(diffs)
        stop_g, creep_g = bands(gt_diffs)
        err = (diffs - gt_diffs).reshape(-1, D)
        pose_err = (chunk - targets.double()).reshape(-1, D)

        # Rotation flips: adjacent WITHIN-CHUNK tick pairs where both ticks rotate past the
        # deadband and the commanded direction reverses. The deadband is a SPEED, so it is
        # scaled by this corpus's tick duration -- identical to `ar_action_head`'s.
        band = FLIP_DEADBAND_RADPS * float(getattr(self, "dt_native", DEFAULT_DT_NATIVE))
        dth = diffs[..., 2]
        active = dth.abs() > band
        both = active[:, :-1] & active[:, 1:]
        flips = ((torch.sign(dth[:, :-1]) * torch.sign(dth[:, 1:])) < 0) & both

        return {
            "sum_sq_err": err.pow(2).sum(0).float(),
            "sum_abs_err": err.abs().sum(0).float(),
            "n_rows": torch.tensor(diffs.shape[0] * diffs.shape[1], device=dev),
            "sum_pose_sq_err": pose_err.pow(2).sum(0).float(),
            "sum_pose_abs_err": pose_err.abs().sum(0).float(),
            "sum_stop_pred": stop_p.float(),
            "sum_stop_gt": stop_g.float(),
            "sum_creep_pred": creep_p.float(),
            "sum_creep_gt": creep_g.float(),
            "sum_flips": flips.double().sum().float(),
            "n_flip_pairs": both.double().sum().float(),
        }

    # -- warm starting the encoders, and nothing else ------------------------------------
    def init_modality_from(self, checkpoint_dir: Union[str, Path]) -> List[str]:
        """Initialise **only** the modality encoders from another run's checkpoint.

        Narrower than `warm_start`, deliberately. `warm_start` borrows every module the
        source checkpoint and this model share -- head, normalizer, adapter -- which across
        HEADS is meaningless: v7's readout MLP emits 30 numbers and this one emits a
        context vector, and its adapter was trained against a Huber objective on that
        readout. The encoders are the one module whose meaning is head-independent: they
        map a pose to a residual on a token embedding, and both runs use the same
        `pose_spec_planar.json` and the same 2048-wide backbone embedding.

        So this loads `blob["modality"]["encoders"]` and touches nothing else -- not the
        head, not the normalizer (which for this head holds the velocity field), not the
        stop head, not the LoRA adapter. Everything else stays at fresh init.

        Strictness is `load_state_blob`'s, which is strict in BOTH directions: weights for
        a spec this run does not declare, a declared spec with no weights in the blob, and
        shape drift all raise. Nothing is skipped silently -- the failure mode this guards
        against is a run that reports "warm started" and trains a randomly-initialised
        encoder, which is indistinguishable from "the injection did not help".

        WHAT IT COSTS. A zero-initialised encoder output makes step 0 invariant to the
        injected values, which is what makes a pose run's step 0 identical to a no-pose
        baseline's. A trained encoder gives that up: from step 0 the pose values move the
        loss. That is a deliberate trade of comparability for convergence speed, not a free
        win -- and it is one the SPEC may already have made, which is worth checking before
        attributing it here: `PlanarSE2Encoder`'s `gain_init` defaults to 0.0 but
        `pose_spec_planar.json` sets it to 1.0, so a fresh encoder under that spec is
        already live at step 0. Against such a spec what the warm start actually costs is
        agreement with a fresh-encoder run, not the zero-init guarantee.

        Returns the encoder keys loaded, so the caller can print them.
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not self.modality_embedder:
            raise RuntimeError(
                f"{checkpoint_dir} was given as a modality warm start but this run declares "
                "no modality specs; pass --modality-specs or drop the warm start"
            )
        blob_path = checkpoint_dir / HEAD_WEIGHTS_FILE
        if not blob_path.exists():
            raise FileNotFoundError(f"no {HEAD_WEIGHTS_FILE} in {checkpoint_dir}")
        blob = torch.load(blob_path, map_location="cpu", weights_only=False)
        modality = blob.get("modality")
        if not modality:
            raise RuntimeError(
                f"{blob_path} has no modality encoder weights (keys: {sorted(blob)}); it is "
                "not from a run with --modality-specs, so there is nothing to warm start "
                "the encoders from"
            )
        # Strict in both directions and on shapes. A mismatch raises here rather than
        # leaving a silently-fresh encoder behind.
        self.modality_embedder.load_state_blob(modality)
        return list(self.modality_embedder.keys)

    # -- checkpointing ------------------------------------------------------------------
    def save_pretrained(self, output_dir: Union[str, Path]):
        super().save_pretrained(output_dir)
        path = Path(output_dir) / HEAD_CONFIG_FILE
        meta = json.loads(path.read_text())
        meta["flow_head_version"] = HEAD_VERSION      # so a loader can dispatch on it
        meta["fm_n_ticks"] = self.n_ticks
        meta["fm_context_dim"] = int(self.target_shape[0])
        meta["fm_decoder_kwargs"] = self.decoder.to_config()
        meta["fm_config"] = asdict(self.fm_cfg)
        # WITHOUT THIS A LATENT CHECKPOINT CANNOT BE LOADED AT ALL: `from_pretrained` would
        # build a codec with no split, and `load_trainable` would then reject the saved
        # `latent.*` keys as unexpected. The weights themselves ride in the normalizer's
        # state dict -- including `rotation`, which is a buffer -- so only the shape-deciding
        # facts are recorded here.
        latent = getattr(self.normalizer, "latent", None)
        if latent is not None:
            meta["fm_latent"] = {"dim": latent.dim, "sigma0": latent.sigma0,
                                 "rotated": latent.rotation is not None}
        path.write_text(json.dumps(meta, indent=2))

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: Union[str, Path],
        processor,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
        **overrides,
    ) -> "TurnFlowActionRegressor":
        checkpoint_dir = Path(checkpoint_dir)
        meta = json.loads((checkpoint_dir / HEAD_CONFIG_FILE).read_text())
        model_cfg = ModelConfig(**{**migrate_model_config(meta["model"]), **overrides})
        loss_cfg = LossConfig(**meta["loss"])
        fm_cfg = FlowMatchingConfig(**meta.get("fm_config", {}))
        # Absent for every checkpoint written before the latent existed, which is exactly the
        # deterministic behaviour those checkpoints had. `rotation` is allocated as the
        # identity purely so the buffer has the right shape for `load_trainable` to fill;
        # the identity is orthogonal, so it also passes the constructor's own check.
        latent_meta = meta.get("fm_latent")
        latent_cfg = None
        if latent_meta:
            dim = int(latent_meta["dim"])
            latent_cfg = LatentConfig(
                enabled=True, sigma0=float(latent_meta["sigma0"]),
                rotation=torch.eye(dim) if latent_meta.get("rotated") else None,
            )
        model = cls.build(
            model_cfg, loss_cfg, lora=None, n_ticks=meta["fm_n_ticks"], processor=processor,
            context_dim=meta["fm_context_dim"],
            decoder_kwargs=meta.get("fm_decoder_kwargs"), fm_cfg=fm_cfg, dtype=dtype,
            latent_cfg=latent_cfg,
        )
        model.train_content_len = meta.get("train_content_len")
        adapter_dir = checkpoint_dir / ADAPTER_SUBDIR
        if adapter_dir.exists():
            from peft import PeftModel

            model.backbone = PeftModel.from_pretrained(model.backbone, str(adapter_dir))
            # The backbone was re-wrapped; re-point the modality hooks through the new
            # wrapper, exactly as `TurnVectorRegressor.from_pretrained` does. A no-op with
            # no specs declared.
            model.attach_modality_hooks()
        model.load_trainable(checkpoint_dir, adapter=False)
        if device:
            model.to(device)
        return model.eval()


# ======================================================================================
# Trainer: the same band-statistic bookkeeping pattern, minus the free-running family
# ======================================================================================
FLOW_METRIC_KEYS = ("sum_stop_pred", "sum_stop_gt", "sum_creep_pred", "sum_creep_gt",
                    "sum_pose_sq_err", "sum_pose_abs_err", "sum_flips", "n_flip_pairs")

#: The CVAE diagnostics, drained the same way and absent unless a latent is installed. Named
#: separately so a reader can see at a glance which keys exist only in a latent run.
LATENT_METRIC_KEYS = ("sum_kl_nats", "sum_sigma", "sum_delta_mu_norm",
                      "sum_delta_mu_over_sigma", "sum_active_dims", "sum_h_std_perdim")


class FlowMatchingSFTTrainer(TurnVectorSFTTrainer):
    """`TurnVectorSFTTrainer` plus the band / composed-pose / flip statistics, all-reduced
    and logged. Structurally the same bookkeeping as `ar_action_head.ARActionSFTTrainer`,
    but NOT a subclass of it: that class's key list and `_drain_metrics` are built around the
    teacher-forced-plus-free-running split, which this head does not have. Subclassing it
    would inherit `free_*` keys that can never be populated here and would invite exactly the
    cross-head comparison the module docstring warns about.

    `rmse_*`/`mae_*` come from the parent, off `sum_sq_err`/`n_rows`, and are per-tick
    DIFFERENTIAL errors. `pose_rmse_*`/`pose_mae_*` are added here and are COMPOSED pose
    errors -- the ones to compare against the AR head's `free_rmse_*`.
    """

    #: Emit the per-dimension motion-band breakdown (`stop_*`, `creep_*`, `near_zero_*`).
    #: Off by default: ~30 keys per log line, built to compare a discrete head's codebook
    #: occupancy against a continuous head's, which is not a question the current mixture
    #: asks. The sums are still ACCUMULATED either way, so enabling this costs only the
    #: logging and nothing downstream loses the ability to compute them.
    emit_motion_bands: bool = False

    def _accumulate(self, outputs: Dict[str, torch.Tensor]):
        super()._accumulate(outputs)
        for key in FLOW_METRIC_KEYS + LATENT_METRIC_KEYS:
            if key not in outputs:
                continue
            v = outputs[key].detach().float()
            self._sums[key] = v.clone() if key not in self._sums else self._sums[key] + v

    def _drain_metrics(self, prefix: str = "") -> Dict[str, float]:
        sums = dict(self._sums)
        out = super()._drain_metrics(prefix)          # turn_loss, rmse_*/mae_*, token counts
        if not out or "n_rows" not in sums:
            return out
        rows = float(sums["n_rows"].clamp(min=1))
        names = DIM_NAMES

        # OFF by default. These per-dimension motion-band breakdowns (`stop_*`, `creep_*`,
        # `near_zero_*`) were built to compare a discrete head's codebook occupancy against a
        # continuous head's, back when the near-zero mass was the thing in question. With the
        # current mixture they are ~30 extra keys per log line that nobody reads, and they
        # crowd out the metrics that are actually consulted (`turn_loss`, `rmse_*`,
        # `pose_rmse_*`). Kept, not deleted -- set `emit_motion_bands = True` on the trainer
        # (`--log-motion-bands`) to get them back for a head comparison.
        if self.emit_motion_bands:
            for key, label in (("sum_stop_pred", "stop_pred"), ("sum_stop_gt", "stop_gt"),
                               ("sum_creep_pred", "creep_pred"), ("sum_creep_gt", "creep_gt")):
                if key not in sums:
                    continue
                for name, v in zip(names, sums[key].tolist()):
                    out[f"{prefix}{label}_{name}"] = v / rows

        # The comparable near-zero statistic. `stop_pred_*` alone is not comparable between a
        # discrete head (whose codebook has a centroid exactly on zero, so it clears
        # EXACT=1e-4 trivially) and a continuous one; the combined mass below the CREEP edge
        # is expressible by both. See the module docstring's section 6 discussion.
        if self.emit_motion_bands and "sum_stop_pred" in sums and "sum_creep_pred" in sums:
            near_p = sums["sum_stop_pred"] + sums["sum_creep_pred"]
            near_g = sums["sum_stop_gt"] + sums["sum_creep_gt"]
            for name, p, g in zip(names, near_p.tolist(), near_g.tolist()):
                out[f"{prefix}near_zero_pred_{name}"] = p / rows
                out[f"{prefix}near_zero_gt_{name}"] = g / rows
            if float(near_g[0]) > 0:
                out[f"{prefix}near_zero_ratio_dx"] = float(near_p[0] / near_g[0])

        # The CVAE diagnostics. Absent -- not zero -- when no latent is installed, so a
        # deterministic run's log line is exactly what it always was.
        for key in LATENT_METRIC_KEYS:
            if key in sums:
                out[f"{prefix}{key[4:]}"] = float(sums[key]) / rows

        if "sum_pose_sq_err" in sums:
            rmse = (sums["sum_pose_sq_err"] / rows).sqrt()
            mae = sums["sum_pose_abs_err"] / rows
            for name, r, m in zip(names, rmse.tolist(), mae.tolist()):
                out[f"{prefix}pose_rmse_{name}"] = r
                out[f"{prefix}pose_mae_{name}"] = m
            out[f"{prefix}pose_rmse_mean"] = float(rmse.mean())

        if "n_flip_pairs" in sums and float(sums["n_flip_pairs"]) > 0:
            out[f"{prefix}rotation_flip"] = float(sums["sum_flips"] / sums["n_flip_pairs"])
        return out
