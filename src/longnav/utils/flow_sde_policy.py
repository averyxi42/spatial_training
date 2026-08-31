"""Flow-SDE policy head: RL over the denoising chain of the SFT flow-matching head.

Design and every decision's evidence: `docs/FLOW_SDE_RL.md`. The one-paragraph version: the
probability-flow ODE and its SDE share marginals, so sampling stochastically reproduces the
action distribution the SFT model already defines and RL initialises AT the SFT policy. The
sampler is hybrid -- of the `K` denoising steps, `N` (default 1-3) are Euler-Maruyama SDE steps
at positions drawn uniformly per chunk, the rest plain ODE steps. Only the `N` stochastic
transitions carry a Gaussian density; their sum is the policy log-prob; one env-step advantage
multiplies it uniformly (no inner GAE, no inner discounting). Eval and deployment run the pure
ODE (`N = 0`) and never see this module's stochastic path.

The head satisfies the same seams `LatentIntentHead` uses, so the existing continuous RL path
needs only capability dispatch (`getattr(head, "chain_log_prob_batch", None)`), never a new
`policy_head.type`:

    forward(hidden)        -> {"h": readout}          # the dict passthrough at
                                                      # vlm_worker.py:644 / :888 carries any dict
    sample_chain_np(h)     -> chain, positions, logprob, chunk    # rollout sampling
    decode_action(chain)   -> (gap, 3)                # the existing actuator seam
    chain_log_prob_batch   -> (B, S) fp32             # old_log_prob recompute AND rl_loss

THE INVARIANT THIS FILE IS ORGANISED AROUND: `sample_chain_np` and `chain_log_prob_batch` must
use the identical transition density, or the PPO ratio is silently wrong. Both therefore call
ONE function, `_sde_transition`, and neither contains its own copy of the math. A convention
mismatch between sampler and scorer is thereby a compile-time impossibility rather than a test
obligation.

Sign conventions are the codebase's own (`euler_integrate`, flow_matching_head.py:349): t = 1 is
noise, `dt = -1/K`, `x <- x + dt * v`, and the field was trained on `u_t = noise - actions` over
`x_t = t * noise + (1 - t) * actions`. Under that convention (derived, not copied -- the
external spec warns the formulas flip with the convention):

    eps_hat = x_t + (1 - t) * v          # exact for the trained target: substitute and check
    score   = -eps_hat / max(t, t_min)   # grad log p_t for the conditional path N((1-t)a, t^2)
    drift   = v - (sigma_t^2 / 2) * score   # reverse-time SDE, dt < 0
    mu      = x_t + dt * drift           # dt NEGATIVE -- same sign as the ODE update
    std     = sigma_t * sqrt(|dt|)

with `sigma_t = a * sqrt(t / max(1 - t, |dt|))` -- the denominator floor is RLinf's
first-step rule (sigma_0 = a*sqrt(K), so first-step injected noise is exactly `a` at any
K). The score's 1/t at the t->0 end CANCELS inside the drift (sigma^2*score =
-a^2*eps_hat/(1-t), shrinking toward t=0), so `n_exclude_last` is a variance choice, not
a singularity guard -- see docs/FLOW_SDE_RL.md "Audit corrections".

Numerics, each a recorded failure mode (docs/FLOW_SDE_RL.md "Failure modes"):
  * the velocity net carries dropout=0.1 and RL runs under `model.train()`; every entry point
    here pins the decoder to eval for its duration, so both recomputes and the rollout sample
    describe the same policy (the `LatentIntentHead.decode_action` precedent);
  * all log-prob math is float32 regardless of model dtype -- a bf16 sum of ~60-180 terms
    cannot resolve the fractions of a nat the ratio needs;
  * `z_0`'s density is parameter-free and cancels in the ratio: stored (the chain layout keeps
    it as block 0, and the first transition needs it) but never summed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from longnav.utils.bin_codec import compose_chunk
# The RTC single construction site (docs/RTC_TRAINING.md): mask, per-tick time and row
# pinning come from the SAME helpers the SFT loss uses -- never re-derived here.
from longnav.utils.flow_matching_head import (
    pin_prefix,
    prefix_mask_from_len,
    prefix_time,
)

HEAD_CONFIG_FILE = "turn_vector_head_config.json"
HEAD_WEIGHTS_FILE = "turn_vector_head.pt"


# ======================================================================================
# Configuration
# ======================================================================================
@dataclass(frozen=True)
class SDEConfig:
    """The sampler's knobs. `n` and `noise_a` are THE two hyperparameters; the rest are
    guards with defaults chosen in docs/FLOW_SDE_RL.md and not expected to move.

    `noise_a` has no safe default: it is the exploration scale, it couples to the learning
    rate (~1/sigma^2), and its usable range is bounded on BOTH sides (fidelity above,
    behavioural indistinguishability below -- the z_0 scatter on this checkpoint is 0.737 rad,
    so too-small `a` explores less than the environment's own noise). It must come from the
    noise sweep, not from a constructor default.
    """

    n: int                      # stochastic steps per chunk, 1 <= n <= K - n_exclude_last
    noise_a: float              # sigma_t = noise_a * sqrt(t / (1 - t))
    # VARIANCE CHOICE, NOT A SINGULARITY GUARD (audited 2026-08-14): the 1/t in the
    # score cancels exactly against sigma_t^2 -- (sigma^2/2)*score = -a^2*eps_hat/
    # (2(1-t)), monotonically SHRINKING toward t=0; the last position is the most benign
    # and terminal marginals are identical for exclude 0..3. The singular end is t=1,
    # handled by the schedule's denominator floor. 0 matches RLinf's default.
    n_exclude_last: int = 1
    t_min: float = 1e-3         # floor inside the score, belt on top of n_exclude_last
    t_sched_min: float = 1e-3   # sigma_t schedule clamp, low side
    t_sched_max: float = 0.95   # ...high side: position 0 sits at t = 1.0 where the schedule
    #                             is singular. 0.95 is half a step below at K = 10 --
    #                             deliberate, config-visible, and validated by the marginal-
    #                             preservation test rather than derived.
    # Per-transition weight on the summed log-prob, a DELIBERATE estimator change in the
    # `logprob_reduction: mean` tradition (biased surrogate, chosen and named, never
    # silent). "none" is the plain joint density. "sigma" multiplies transition k's
    # log-prob by sigma_k / sigma_0 (<= 1): on-policy the density gradient scales as
    # 1/sigma_k, so late low-noise refinement steps otherwise dominate every update ~10x
    # (measured: decoder grad x1.0 -> x9.6 across positions, tracking 1/sigma to within
    # 10%). The weight equalizes per-position gradient scale -- equivalently a per-position
    # learning rate proportional to sigma_k -- at 0.914 direction overlap with the
    # unweighted gradient. Applied identically in the sampler's reported log-prob and the
    # scorer, so the ratio stays a ratio of one consistent quantity.
    position_weight: str = "none"

    def __post_init__(self):
        if self.position_weight not in ("none", "sigma"):
            raise ValueError(
                f"position_weight must be 'none' or 'sigma', got {self.position_weight!r}")
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n} (use the plain ODE for n = 0)")
        if not (self.noise_a > 0.0):
            raise ValueError(
                f"noise_a must be positive, got {self.noise_a}. There is no default: it is "
                "the exploration scale and must come from the noise sweep "
                "(docs/FLOW_SDE_RL.md, Validation)."
            )
        if self.n_exclude_last < 0:
            raise ValueError(f"n_exclude_last must be >= 0, got {self.n_exclude_last}")


# ======================================================================================
# The one transition density (the sampler/scorer invariant lives HERE)
# ======================================================================================
def _sde_transition(decoder: nn.Module, ctx: torch.Tensor, x_t: torch.Tensor,
                    t: torch.Tensor, dt: float, cfg: SDEConfig,
                    prefix_mask: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """One SDE step's Gaussian: `(mu, std)` of `z_next` given `(ctx, x_t, t)`.

    Everything is float32 on entry and exit. `t` is per-row `(B,)` -- the batched scorer calls
    this with different positions per row and the decoder already takes vector time.

    The score is NOT detached: it contains the velocity net, and detaching it silently changes
    the gradient estimator into a different algorithm (external spec, wiring rule 2).
    `sigma_t` scales the drift correction (as sigma^2/2) and the injected noise (as
    sigma*sqrt|dt|) from the SAME tensor -- the two are halves of one identity and any knob
    that scales one without the other is wrong (wiring rule 1).

    `prefix_mask` (`(B, T)` bool, True at RTC-committed rows) makes the transition on
    those rows a DIRAC: the field sees per-tick flow time 0 there (this repo's reversed
    axis -- clean, see docs/RTC_RL.md section 2), and `mu` is pinned to `x_t`, so the
    committed rows carry no transition randomness. Their density terms are excluded by
    the caller (`_gaussian_logprob(prefix_mask=...)`): prefix rows are STATE, and the
    Gaussian evaluated at a pinned row is a garbage finite number that differs between
    old and new policy. Sampler and scorer both come through here -- the module's one
    invariant -- so the mask lives in this signature, nowhere else.
    """
    x_t = x_t.float()
    t = t.float()
    time_in = prefix_time(t, prefix_mask) if prefix_mask is not None else t
    v = decoder(ctx.float(), x_t, time_in)                              # (B, T, 3)
    tb = t.view(-1, 1, 1)
    eps_hat = x_t + (1.0 - tb) * v
    # t_min cannot bind for any K <= 1000 (source times are >= 1/K); if it ever does,
    # DESTINATION times are being fed to the scorer -- treat a binding clamp as an
    # off-by-one detector, not a guard (independent audit, 2026-08-14).
    score = -eps_hat / tb.clamp_min(cfg.t_min)
    sigma_t = _sigma(t, cfg, dt).view(-1, 1, 1)
    # SIGN: reverse-time drift is v MINUS (sigma^2/2)*score under this codebase's
    # t:1->0, dt<0 convention (piRL Eq. 8; verified against an analytic Gaussian flow
    # where the true terminal marginal is known -- the "+" sign converges to a ~4x wider
    # distribution, with distortion growing as a^2, which is exactly why the low-a
    # marginal-preservation test never caught it). See tests/test_flow_sde_policy.py::
    # test_high_noise_marginal_preservation for the guard.
    mu = x_t + dt * (v - 0.5 * sigma_t.pow(2) * score)
    if prefix_mask is not None:
        mu = torch.where(prefix_mask[..., None], x_t, mu)   # Dirac on committed rows
    std = sigma_t * math.sqrt(abs(dt))
    return mu, std.expand_as(mu)


def _sigma(t: torch.Tensor, cfg: SDEConfig, dt: float) -> torch.Tensor:
    """THE noise schedule -- the only implementation; sampler, scorer and position
    weights all read it here (the module's one-function invariant applies to the
    schedule too; a second copy silently desynchronised once).

    sigma_t = a * sqrt(t / max(1 - t, |dt|)). The denominator floor is RLinf's first-step
    rule: at t = 1 (position 0) it gives sigma_0 = a*sqrt(K), so the first step's
    INJECTED noise sigma_0*sqrt|dt| equals `a` exactly, at every K -- which is what makes
    `a` mean the same thing as piRL's swept `a` (their working point a~0.5, failure
    a~0.2). The previous t<=0.95 clamp made first-step noise a*sqrt(19/K):
    K-dependent, 1.38x RLinf's at K=10, and not transferable across K. Any bounded
    schedule is marginal-preserving (the drift and diffusion share this tensor); the
    choice is about comparability, not correctness."""
    t = t.float()
    denom = (1.0 - t).clamp_min(abs(dt))
    return cfg.noise_a * torch.sqrt(t.clamp_min(0.0) / denom)


def _gaussian_logprob(x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor,
                      prefix_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Sum over the chunk dims, keep the batch dim. Float32 throughout.

    `prefix_mask` excludes RTC-committed rows from the sum, as a MULTIPLIER: their
    transition is a Dirac (see `_sde_transition`), and even at a pinned row where
    `x == mu` the Gaussian contributes `-0.5*log(2*pi*var)` per element -- an
    `h`-independent constant per transition, but one the masked objective must not
    carry (docs/RTC_RL.md section 2)."""
    var = std.pow(2)
    lp = -0.5 * ((x - mu).pow(2) / var + torch.log(2.0 * math.pi * var))
    if prefix_mask is not None:
        lp = lp * (~prefix_mask)[..., None].to(lp.dtype)
    return lp.flatten(1).sum(dim=1)


def _position_weight(k: torch.Tensor, cfg: SDEConfig, K: int) -> torch.Tensor:
    """`(rows,)` weight for transition index `k` under `cfg.position_weight`.

    For "sigma": sigma_k / sigma_0 with both taken from the clamped schedule, so
    `noise_a` cancels and the weight depends only on (k, K, clamps). Position 0 always
    weighs 1.0 and the weight is monotonically decreasing in k."""
    if cfg.position_weight == "none":
        return torch.ones_like(k, dtype=torch.float32)
    dt = -1.0 / K
    t_k = 1.0 + k.float() * dt
    sig = _sigma(t_k, cfg, dt)
    # ones_like keeps the device (a CPU-constructed sig0 crashed the first GPU run --
    # the unit suite exercises this path on CPU only).
    sig0 = _sigma(torch.ones_like(t_k[:1], dtype=torch.float32), cfg, dt)
    return sig / sig0


class _decoder_eval:
    """Pin the velocity net to eval() for the duration, restoring the caller's mode.

    Inside the head's methods rather than at any call site, so the guarantee is
    caller-independent -- the exact remedy `LatentIntentHead.decode_action` uses, needed
    doubly here because dropout in a RECOMPUTED log-prob corrupts the PPO ratio rather than
    just an action."""

    def __init__(self, decoder: nn.Module):
        self.decoder = decoder

    def __enter__(self):
        self.was_training = self.decoder.training
        self.decoder.eval()

    def __exit__(self, *exc):
        self.decoder.train(self.was_training)
        return False


# ======================================================================================
# The head
# ======================================================================================
class FlowSDEHead(nn.Module):
    """`hidden_states -> {"h"}`, plus the chain sampler/scorer/actuator.

    `forward` deliberately returns only the readout: the policy's distribution is over the
    chain, which no `(mu, log_std)` pair can describe, and anything downstream that needs the
    density calls `chain_log_prob_batch`. The dict passthrough carries `{"h"}` untouched;
    the capability checks (`sample_chain_np`, `chain_log_prob_batch`) are how the three
    branch sites recognise this head -- the actuator-seam idiom, not a type string.
    """

    def __init__(self, readout: nn.Module, codec: nn.Module, gap: int, sde: SDEConfig):
        super().__init__()
        if int(gap) < 1:
            raise ValueError(f"gap must be >= 1, got {gap}")
        self.readout = readout
        self.codec = codec
        self.gap = int(gap)
        self.sde = sde
        dec = codec.decoder
        self.K = int(codec.num_inference_steps)
        self.n_ticks, self.n_dims = int(dec.n_ticks), int(dec.n_dims)
        if sde.n > self.K - sde.n_exclude_last:
            raise ValueError(
                f"n={sde.n} stochastic steps but only {self.K - sde.n_exclude_last} "
                f"admissible positions (K={self.K}, n_exclude_last={sde.n_exclude_last})."
            )
        # The rollout draws through a private generator so a seeded run is reproducible
        # without touching the global stream (the latent head's `latent_generator` pattern).
        self._gen: Optional[torch.Generator] = None
        # When True, sample_chain_np runs the PURE ODE (no stochastic transitions,
        # logprob 0) -- the eval/deploy sampler, toggled per-cycle by interleaved eval.
        self.force_ode = False

    # -- shapes ---------------------------------------------------------------------
    @property
    def block(self) -> int:
        return self.n_ticks * self.n_dims                      # floats per chain element

    @property
    def chain_len(self) -> int:
        return (self.K + 1) * self.block                       # z_0 .. z_K, flattened

    def seed(self, seed: int) -> None:
        dev = next(self.parameters()).device
        self._gen = torch.Generator(device=dev)
        self._gen.manual_seed(int(seed))

    # -- the policy_stats contract ----------------------------------------------------
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, _ = hidden_states.shape
        flat = hidden_states.reshape(b * t, 1, hidden_states.shape[-1])
        h = self.readout(flat)                                  # (B*T, ctx)
        return {"h": h.reshape(b, t, -1).float()}

    # -- rollout: sample -------------------------------------------------------------
    @torch.no_grad()
    def sample_chain_np(self, h: np.ndarray, prefix: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """One chunk: `(chain, positions, logprob, chunk)`.

        `chain` is `((K+1) * T * 3,)` float32 -- z_0 through z_K flattened, so what is stored
        for the ratio is byte-identical to what produced the executed action. `positions` is
        `(n,)` int64, ascending; transition k maps chain block k to block k+1. `logprob` is
        the summed density of the `n` stochastic transitions only.

        `prefix` is the RTC commitment (docs/RTC_RL.md section 2): `(d, 3)` PHYSICAL
        per-tick differentials, `d` possibly 0. Passing it -- even empty -- declares the
        RTC contract: prefix rows are pinned clean at per-tick flow time t = 0 (this
        repo's REVERSED axis: t = 1 noise, t = 0 data) through the whole integration,
        and the returned `chunk` is the FULL `(n_ticks, 3)` block -- the env owns the
        slicing, because the tail is the next commitment's source. `prefix=None` is the
        historical contract exactly: unconditioned chain, `(gap, 3)` truncated chunk.

        The stochastic (RL) sampler conditions through `_sde_transition(prefix_mask=)`:
        committed rows are Dirac (mu pinned, field at per-tick t = 0), their density
        terms are excluded, and the chain blocks carry the pinned rows so the scorer's
        constancy assert can hold them against the stored prefix (docs/RTC_RL.md
        section 2).
        """
        dev = next(self.parameters()).device
        ctx = torch.as_tensor(np.asarray(h, np.float32), device=dev).reshape(1, -1)
        cfg, K, dt = self.sde, self.K, -1.0 / self.K
        rtc = prefix is not None
        prefix_full = prefix_mask = time_of = None
        if rtc:
            p = torch.as_tensor(np.asarray(prefix, np.float64),
                                device=dev).reshape(-1, self.n_dims)
            d = p.shape[0]
            prefix_mask = prefix_mask_from_len(
                torch.full((1,), d, dtype=torch.long, device=dev), self.n_ticks)
            prefix_full = torch.zeros(1, self.n_ticks, self.n_dims,
                                      device=dev, dtype=torch.float32)
            if d:
                prefix_full[:, :d] = self.codec.scale(p).float()
            time_of = lambda t_k: prefix_time(t_k, prefix_mask)   # noqa: E731
        admissible = K - cfg.n_exclude_last
        if self._gen.device != dev:
            # The generator was made at construction (CPU) and the head moved since;
            # re-home it on the parameters' device, keeping its seed.
            g = torch.Generator(device=dev)
            g.manual_seed(int(self._gen.initial_seed()))
            self._gen = g
        perm = torch.randperm(admissible, generator=self._gen, device=dev)[: cfg.n]
        positions = perm.sort().values
        if self.force_ode:
            # Empty positions, not just an empty pos_set: an ODE chain carrying
            # stochastic positions could be handed to the scorer and yield a finite,
            # confidently wrong log-prob. Make that structurally impossible.
            positions = positions[:0]
        pos_set = set(positions.tolist())

        with _decoder_eval(self.codec.decoder):
            x = torch.randn(1, self.n_ticks, self.n_dims, device=dev,
                            dtype=torch.float32, generator=self._gen)
            if rtc:
                x = pin_prefix(x, prefix_full, prefix_mask)
            blocks = [x]
            logprob = torch.zeros(1, device=dev)
            for k in range(K):
                t_k = torch.full((1,), 1.0 + k * dt, device=dev)
                if k in pos_set:
                    mu, std = _sde_transition(self.codec.decoder, ctx, x, t_k, dt, cfg,
                                              prefix_mask=prefix_mask)
                    eps = torch.randn(mu.shape, device=dev, dtype=torch.float32,
                                      generator=self._gen)
                    x = mu + std * eps
                    if rtc:
                        # Re-pin BEFORE the density: the stored block must carry the
                        # exact committed rows, and the density is over postfix
                        # elements of exactly the stored block.
                        x = pin_prefix(x, prefix_full, prefix_mask)
                    w = _position_weight(torch.tensor([k], device=dev), cfg, K)
                    logprob = logprob + w * _gaussian_logprob(x, mu, std,
                                                              prefix_mask=prefix_mask)
                else:
                    v = self.codec.decoder(ctx, x, time_of(t_k) if rtc else t_k)
                    x = x + dt * v
                    if rtc:
                        x = pin_prefix(x, prefix_full, prefix_mask)
                blocks.append(x)

        chain = torch.cat([b.reshape(1, -1) for b in blocks], dim=1)[0]
        chunk = self._compose(blocks[-1], full=rtc)
        return (chain.cpu().numpy().astype(np.float32),
                positions.cpu().numpy().astype(np.int64),
                float(logprob.item()),
                chunk)

    # -- the actuator seam (rollout_core.py:204, unchanged) ----------------------------
    @torch.no_grad()
    def decode_action(self, action: np.ndarray) -> np.ndarray:
        """Stored flat chain -> the `(gap, 3)` chunk the environment executes.

        The action IS the chain; the executed chunk is its last block, unscaled and composed
        exactly as `denormalize_from_latent` composes -- then truncated to `gap` rows, which
        is a prefix of the same trajectory because the chunk is cumulative anchor-relative."""
        dev = next(self.parameters()).device
        flat = torch.as_tensor(np.asarray(action, np.float32), device=dev)
        z_K = flat[-self.block:].reshape(1, self.n_ticks, self.n_dims)
        return self._compose(z_K)

    def _compose(self, z_K: torch.Tensor, full: bool = False) -> np.ndarray:
        chunk = compose_chunk(self.codec.unscale(z_K.double()))          # (1, T, 3)
        if full:
            # The RTC contract: the env owns the slicing (the tail is the next
            # commitment's source), so the head hands over every row.
            return chunk[0].float().cpu().numpy()
        return chunk[0, : self.gap].float().cpu().numpy()

    # -- training: score stored transitions under current theta ------------------------
    def chain_log_prob_batch(self, h: torch.Tensor, chains: torch.Tensor,
                             positions: torch.Tensor,
                             prefix_actions: Optional[torch.Tensor] = None,
                             prefix_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`(B, S)` summed log-probs, differentiable, float32.

        `h` is `(B, S, ctx)` (the `policy_stats["h"]` of a training forward), `chains` is
        `(B, S, chain_len)` -- the STORED latents; `positions` is `(B, S, n)`.

        Stored latents are data. The deterministic prefix is NOT re-integrated under the
        current parameters -- re-integrating would score a different chain than the one that
        acted, which changes the estimator (docs/FLOW_SDE_RL.md, storage). One decoder call
        per stochastic slot scores the whole batch: the decoder takes vector time, so rows
        with different positions batch together.

        RTC (docs/RTC_RL.md sections 2 and 5): `prefix_actions` is `(B, S, d_max, 3)`
        PHYSICAL differentials, zero-padded; `prefix_len` is `(B, S)`. Both come from
        the rollout buffer, NEVER re-derived from poses -- the 1/sigma^2-amplified
        density does not forgive a compose/decompose round trip. The scorer rebuilds
        the mask, reconditions `_sde_transition` identically to the sampler, and
        ASSERTS the stored chain's committed rows equal the stored prefix (a broken
        pin would otherwise score a chain that never acted).
        """
        B, S = chains.shape[0], chains.shape[1]
        n, dt = self.sde.n, -1.0 / self.K
        rows = B * S
        ctx = h.reshape(rows, -1).float()
        ch = chains.reshape(rows, self.K + 1, self.n_ticks, self.n_dims).float()
        pos = positions.reshape(rows, n).long()

        mask = None
        if prefix_actions is not None:
            if prefix_len is None:
                raise ValueError("prefix_actions without prefix_len")
            plen = prefix_len.reshape(rows).long()
            # prefix_mask_from_len raises on d >= n_ticks -- the loud guard, kept: a
            # violating stored d means the buffer is corrupt, not that scoring should
            # improvise a mask.
            mask = prefix_mask_from_len(plen, self.n_ticks)
            pa = prefix_actions.reshape(rows, -1, self.n_dims)
            scaled = self.codec.scale(pa.double()).float()
            pf_full = torch.zeros(rows, self.n_ticks, self.n_dims,
                                  device=ch.device, dtype=torch.float32)
            pf_full[:, : scaled.shape[1]] = scaled
            drift = ((ch[:, 0] - pf_full).abs() * mask[..., None].float()).max()
            if float(drift) > 1e-4:
                raise RuntimeError(
                    f"stored chain's committed rows drift {float(drift):.2e} from the "
                    "stored prefix_actions: the sampler's pin is broken or the buffer "
                    "was corrupted -- scoring would rate a chain that never acted"
                )

        with _decoder_eval(self.codec.decoder):
            total = torch.zeros(rows, device=ctx.device, dtype=torch.float32)
            for j in range(n):
                k = pos[:, j]                                            # (rows,)
                t_k = 1.0 + k.float() * dt
                idx = torch.arange(rows, device=ctx.device)
                x_k = ch[idx, k]                                         # (rows, T, 3)
                x_next = ch[idx, k + 1]
                mu, std = _sde_transition(self.codec.decoder, ctx, x_k, t_k, dt,
                                          self.sde, prefix_mask=mask)
                w = _position_weight(k, self.sde, self.K)
                total = total + w * _gaussian_logprob(x_next, mu, std, prefix_mask=mask)
        return total.reshape(B, S)

    # -- run-log honesty ---------------------------------------------------------------
    def describe(self) -> str:
        return (f"flow-SDE head: K={self.K} n={self.sde.n} a={self.sde.noise_a} "
                f"exclude_last={self.sde.n_exclude_last} pos_weight={self.sde.position_weight} "
                f"gap={self.gap} chain={self.chain_len} floats  (eval/deploy = pure ODE)")

    # -- construction ------------------------------------------------------------------
    @classmethod
    def from_policy_head_config(cls, cfg: Dict[str, Any], input_dim: int,
                                dtype: torch.dtype) -> "FlowSDEHead":
        ckpt = cfg.get("checkpoint_dir")
        if not ckpt:
            raise ValueError(
                "flow_sde head requires `checkpoint_dir`: the readout and the velocity "
                "field are trained modules, and a fresh one is not a policy."
            )
        readout, codec, _ = load_flow_stack(ckpt, dtype=dtype)
        head = cls(
            readout=readout, codec=codec, gap=int(cfg["gap"]),
            sde=SDEConfig(
                n=int(cfg.get("sde_n", 1)),
                noise_a=float(cfg["sde_noise_a"]),
                n_exclude_last=int(cfg.get("sde_exclude_last", 1)),
                position_weight=str(cfg.get("sde_position_weight", "none")),
            ),
        )
        seed = cfg.get("sde_seed")
        if seed is not None:
            head.seed(int(seed))
        _ = input_dim   # signature parity with ContinuousActionHead
        return head


# ======================================================================================
# Loading (no backbone, latent optional)
# ======================================================================================
def load_flow_stack(checkpoint_dir, dtype: torch.dtype = torch.float32):
    """`(readout, codec, context_dim)` without instantiating the backbone.

    Unlike `load_latent_stack` this does NOT require `fm_latent`: the flow-SDE policy's
    natural starting point is the DETERMINISTIC checkpoint (that is the entire point -- RL
    initialises at the strongest SFT policy). A latent checkpoint still loads -- the split is
    reconstructed so the strict state-dict load passes -- and its latent is simply never used:
    this head conditions the decoder on `h` directly.
    """
    from longnav.utils.flow_matching_head import FlowActionCodec, FlowActionDecoder
    from longnav.utils.latent_intent import LatentSplit
    from longnav.utils.turn_vectors import TurnVectorHead

    d = Path(checkpoint_dir)
    meta = json.loads((d / HEAD_CONFIG_FILE).read_text())
    model_cfg, ctx_dim = meta["model"], int(meta["fm_context_dim"])

    readout = TurnVectorHead(
        hidden_size=int(meta.get("backbone_hidden_size") or model_cfg.get(
            "backbone_hidden_size") or 2048),
        out_dim=ctx_dim,
        mode=model_cfg.get("pool_mode", "mean"),
        content_len=1 if model_cfg.get("pool_mode") == "flat" else None,
        hidden_dims=tuple(model_cfg.get("head_hidden_dims", ())),
        dropout=float(model_cfg.get("head_dropout", 0.0)),
        layer_norm=bool(model_cfg.get("head_layer_norm", True)),
        standardize=bool(model_cfg.get("standardize_head_inputs", False)),
        dtype=dtype,
    )
    decoder = FlowActionDecoder(
        context_dim=ctx_dim, n_ticks=int(meta["fm_n_ticks"]),
        **(meta.get("fm_decoder_kwargs") or {}),
    )
    latent_meta = meta.get("fm_latent")
    latent = None
    if latent_meta:
        dim = int(latent_meta["dim"])
        latent = LatentSplit(
            dim=dim, sigma0=float(latent_meta["sigma0"]),
            rotation=torch.eye(dim) if latent_meta.get("rotated") else None,
        )
    codec = FlowActionCodec(
        decoder,
        num_inference_steps=int((meta.get("fm_config") or {}).get("num_inference_steps", 10)),
        action_scales=(meta.get("fm_config") or {}).get("action_scales", (0.03, 0.03, 0.05)),
        latent=latent,
    )
    blob = torch.load(d / HEAD_WEIGHTS_FILE, map_location="cpu", weights_only=False)
    readout.load_state_dict(blob["head"], strict=True)
    codec.load_state_dict(blob["normalizer"], strict=True)
    return readout.eval(), codec.eval(), ctx_dim
