"""Conditional normalizing flow over per-tick action differentials.

The replacement for the Huber regression head. The regression head is a
conditional-*mean* estimator (Huber with beta=1.0 against targets of magnitude ~0.02 is
plain MSE), and the action data is a stop-or-drive mixture, so its optimum lands in the
gap between the modes -- which is what creeping is. A flow trained by maximum likelihood
models the whole conditional distribution, so it can be multimodal and commit.

What is modelled
----------------
The 10 poses in an `action_chunks` row are all relative to a *single* anchor, so they are
cumulative, heavily correlated and badly conditioned. The flow works on **per-tick
body-frame differentials** instead: 30 dimensions (10 ticks x 3 channels), laid out in
C-order `(tick, channel)` exactly as `cnf_head.data` caches them. That is the space where
"the robot did not move during this tick" is the origin in every coordinate.

Nothing is projected or reduced. `log_prob` is an exact density **in action space**, in
the target's own units (m, m, rad per tick): the per-channel normalisation is a fixed
diagonal affine whose log-determinant is folded in. Any dimensionality reduction would
map into a measure-zero subset of action space and forfeit that, which matters because
RL is downstream (`dump/overnight/RL_INTERFACE.md`).

The one thing that genuinely breaks a flow, and the fix
------------------------------------------------------
The targets contain literal atoms: 34% of ticks have *bitwise* zero forward motion, with
a further ~28% inside a float-residue cloud out to ~1e-5 m left by composing a pose that
did not change (`dump/cnf_head/flow/degeneracy.json`). A continuous density on an atom
has unbounded likelihood -- NLL runs to -inf, gradients explode, log-probs stop meaning
anything.

The remedy is a **noise floor**: train on `x + sigma * eps`. That bounds the density
above by `prod_i 1/(sqrt(2 pi) sigma_i)`, hence bounds NLL below, and the bound is
computed and logged (`min_nll_per_dim`) so divergence past it is immediately visible as a
bug rather than as progress. `sigma = 1e-5` in raw units per channel is the default: it
is the measured width of the data's own residue cloud, and it moves **no** band statistic
at all (forward exact-stop 63.1% -> 63.1%, creep 2.7% -> 2.7% on 62M training ticks),
whereas 1e-4 would move a fifth of the stopped ticks into the creep band and corrupt the
metric the whole project is about. It is annealed from a larger value over the first
steps, because the early optimisation is where a flow blows up.

Strafe is kept. The brief expected it to be near-constant-zero and the worst offender;
measured, its atom is *the same ticks* as forward's (34.2% bitwise zero, because a
stopped robot is stopped in every channel) and away from the atom it carries real,
non-redundant mass -- 23% of ticks in the 0.1 mm - 1 cm band, and a quadratic fit on the
other two channels explains only 6.6% of its variance. It is not degenerate, so dropping
it would cost the exact action-space density for nothing.

Architecture
------------
Affine coupling (RealNVP), conditioned on the VLM turn vector. Small on purpose -- this
is a head, not a backbone. Per layer: ActNorm, then an affine coupling whose scale is
`s_max * tanh(raw / s_max)` so a log-scale can never run away. Masks come in
complementary pairs from a seeded RNG, so every dimension is transformed at least once
per pair.

RL surface
----------
Built in now rather than retrofitted (`dump/overnight/RL_INTERFACE.md`):

    log_prob(x, ctx)                 exact, at arbitrary x, one pass
    sample(ctx) -> (x, log_prob)     reparameterised, both from ONE pass
    entropy(ctx)                     Monte Carlo, unbiased, free while sampling
    mode(ctx) / best_of_k(ctx, k)    seeded deterministic evaluation paths

Note the difference from the superseded autoencoder design: there is no decoder, so RL
operates directly on the action and `log_prob` is a genuine action-space density. The
`RL_INTERFACE.md` operations are all stated in `z` there; here they are in `x`, which is
strictly better and removes the "flat region is behaviourally unidentifiable" pathology
that document was written to warn about.

Torch-only (no transformers/peft), so it imports in both the model environment and the
analysis environment.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

CHUNK_LEN = 10
N_CHANNELS = 3
DIM = CHUNK_LEN * N_CHANNELS

# Raw-unit noise floor per channel (m, m, rad). See the module docstring: this is the
# measured width of the data's float-residue cloud and it is band-statistic invisible.
DEFAULT_NOISE_STD = (1e-5, 1e-5, 1e-5)
# Per-channel scale for normalisation: the 99th percentile of |value| on the train split
# (dump/cnf_head/flow/degeneracy.json -> train.channel_scale_p99). Scaling only, never
# centring -- a mean shift would move the data's zero atom off the origin for no gain.
DEFAULT_CHANNEL_SCALE = (0.0800, 0.0289, 0.0995)


def _tile_channels(v: Sequence[float], chunk_len: int = CHUNK_LEN) -> torch.Tensor:
    """(3,) per-channel -> (30,) per-dimension, C-order (tick, channel)."""
    t = torch.as_tensor(v, dtype=torch.float32).flatten()
    if t.numel() == DIM:
        return t
    assert t.numel() == N_CHANNELS, f"expected 3 or {DIM} values, got {t.numel()}"
    return t.repeat(chunk_len)


# ======================================================================================
# Layers
# ======================================================================================
class ActNorm(nn.Module):
    """Per-dimension affine, `u = (x - bias) * exp(-log_scale)`.

    Initialised from dataset statistics passed in at construction rather than lazily from
    the first batch: lazy init desynchronises under DDP and makes a run depend on which
    conversation happened to be first. Identity by default.
    """

    def __init__(self, dim: int, mean: Optional[torch.Tensor] = None,
                 log_scale: Optional[torch.Tensor] = None):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim) if mean is None else mean.clone())
        self.log_scale = nn.Parameter(
            torch.zeros(dim) if log_scale is None else log_scale.clone())

    def normalise(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = (x - self.bias) * torch.exp(-self.log_scale)
        return u, -self.log_scale.sum().expand(x.shape[0])

    def generate(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = u * torch.exp(self.log_scale) + self.bias
        return x, self.log_scale.sum().expand(u.shape[0])


class AffineCoupling(nn.Module):
    """RealNVP coupling: half the dimensions are passed through and condition the rest.

    The scale is bounded, `s = s_max * tanh(raw / s_max)`. Unbounded log-scales are the
    usual way a flow on near-atomic data dies: the model discovers it can win likelihood
    by concentrating without limit, the log-determinant runs away, and the run is over
    before the loss curve looks wrong. `s_max` caps the concentration any one layer can
    contribute; the stack's total capacity is `n_layers * s_max` nats per dimension,
    against the ~8.9 nats the tightest dimension's noise floor actually needs.
    """

    def __init__(self, dim: int, mask: torch.Tensor, context_dim: int = 0,
                 hidden: int = 256, depth: int = 2, s_max: float = 5.0):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.s_max = float(s_max)
        layers, d_in = [], dim + context_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            d_in = hidden
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(d_in, 2 * dim)
        # Start as the identity: zeroed output layer means s = t = 0 on step 0, so the
        # flow begins as (ActNorm-transformed) noise and the log-determinant starts at a
        # known finite value instead of wherever initialisation happens to put it.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _params(self, x_masked: torch.Tensor, ctx: Optional[torch.Tensor]):
        h = x_masked if ctx is None else torch.cat([x_masked, ctx], dim=-1)
        s, t = self.out(self.net(h)).chunk(2, dim=-1)
        s = self.s_max * torch.tanh(s / self.s_max)
        keep = 1.0 - self.mask
        return s * keep, t * keep

    def normalise(self, x: torch.Tensor, ctx: Optional[torch.Tensor]):
        xa = x * self.mask
        s, t = self._params(xa, ctx)
        u = xa + (1.0 - self.mask) * ((x - t) * torch.exp(-s))
        return u, -s.sum(dim=-1)

    def generate(self, u: torch.Tensor, ctx: Optional[torch.Tensor]):
        ua = u * self.mask
        s, t = self._params(ua, ctx)          # masked half is unchanged, so xa == ua
        x = ua + (1.0 - self.mask) * (u * torch.exp(s) + t)
        return x, s.sum(dim=-1)


def build_masks(dim: int, n_layers: int, seed: int = 0) -> torch.Tensor:
    """Complementary pairs of random half-masks.

    Pairs rather than independent draws: with independent halves a dimension can sit in
    the conditioner side of every layer and never be transformed at all (probability
    2^-n_layers, small but not zero, and silent when it happens). Pairing makes it
    impossible.
    """
    g = torch.Generator().manual_seed(seed)
    masks = []
    for i in range(n_layers):
        if i % 2 == 0:
            m = torch.zeros(dim)
            m[torch.randperm(dim, generator=g)[: dim // 2]] = 1.0
        else:
            m = 1.0 - masks[-1]
        masks.append(m)
    return torch.stack(masks)


# ======================================================================================
# The flow
# ======================================================================================
class ConditionalFlow(nn.Module):
    """p(action chunk | context), as an exact density in the target's own units.

    `x` is always raw `(B, 10, 3)` per-tick differentials in metres/radians -- callers
    never see the normalised space. `ctx` is `(B, context_dim)` or None.
    """

    def __init__(
        self,
        context_dim: int = 0,
        n_layers: int = 12,
        hidden: int = 256,
        depth: int = 2,
        s_max: float = 5.0,
        chunk_len: int = CHUNK_LEN,
        n_channels: int = N_CHANNELS,
        channel_scale: Sequence[float] = DEFAULT_CHANNEL_SCALE,
        noise_std: Sequence[float] = DEFAULT_NOISE_STD,
        actnorm_mean: Optional[Sequence[float]] = None,
        actnorm_log_scale: Optional[Sequence[float]] = None,
        mask_seed: int = 0,
    ):
        super().__init__()
        self.chunk_len, self.n_channels = chunk_len, n_channels
        self.dim = chunk_len * n_channels
        self.context_dim = int(context_dim)
        self.n_layers, self.hidden, self.depth = n_layers, hidden, depth
        self.s_max, self.mask_seed = float(s_max), int(mask_seed)

        scale = _tile_channels(channel_scale, chunk_len)
        self.register_buffer("unit_scale", scale)
        self.register_buffer("noise_std_raw", _tile_channels(noise_std, chunk_len))
        # log|det d(x_norm)/d(x_raw)| -- constant, folded into every log_prob so the
        # density is reported in raw action units and RL never has to remember a Jacobian.
        self.register_buffer("log_unit_det", -torch.log(scale).sum())

        masks = build_masks(self.dim, n_layers, seed=mask_seed)
        mean = None if actnorm_mean is None else _tile_channels(actnorm_mean, chunk_len)
        lsc = (None if actnorm_log_scale is None
               else _tile_channels(actnorm_log_scale, chunk_len))
        self.actnorms = nn.ModuleList(
            [ActNorm(self.dim, mean if i == 0 else None, lsc if i == 0 else None)
             for i in range(n_layers)])
        self.couplings = nn.ModuleList(
            [AffineCoupling(self.dim, masks[i], context_dim=self.context_dim,
                            hidden=hidden, depth=depth, s_max=s_max)
             for i in range(n_layers)])

    # -- units ------------------------------------------------------------------------
    def to_normalised(self, x_raw: torch.Tensor) -> torch.Tensor:
        return x_raw.reshape(x_raw.shape[0], -1) / self.unit_scale

    def to_raw(self, x_norm: torch.Tensor) -> torch.Tensor:
        return (x_norm * self.unit_scale).reshape(-1, self.chunk_len, self.n_channels)

    def dequantise(self, x_raw: torch.Tensor, sigma_mult: float = 1.0,
                   generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Add the noise floor. `sigma_mult` implements the anneal; 0 disables."""
        if sigma_mult <= 0:
            return x_raw
        flat = x_raw.reshape(x_raw.shape[0], -1)
        eps = torch.empty_like(flat).normal_(generator=generator)
        out = flat + eps * (self.noise_std_raw * sigma_mult)
        return out.reshape_as(x_raw)

    def min_nll_per_dim(self, sigma_mult: float = 1.0) -> float:
        """The likelihood floor the noise buys, in nats per dimension.

        `log p <= -sum_i log(sqrt(2 pi) sigma_i)` in normalised units for a density that
        is pure noise-smeared atom, so NLL can never legitimately go below this. Logged
        every step: a run that crosses it is diverging, not learning, and that is worth
        knowing within seconds rather than at the end.
        """
        s = self.noise_std_raw * max(sigma_mult, 1e-12) / self.unit_scale
        return float(torch.log(math.sqrt(2 * math.pi) * s).sum() / self.dim)

    # -- the two directions -----------------------------------------------------------
    def _normalising(self, x_norm: torch.Tensor, ctx: Optional[torch.Tensor]):
        """x -> u. Returns (u, log|det du/dx| in normalised units)."""
        logdet = torch.zeros(x_norm.shape[0], device=x_norm.device, dtype=x_norm.dtype)
        h = x_norm
        for act, cpl in zip(self.actnorms, self.couplings):
            h, ld = act.normalise(h)
            logdet = logdet + ld
            h, ld = cpl.normalise(h, ctx)
            logdet = logdet + ld
        return h, logdet

    def _generating(self, u: torch.Tensor, ctx: Optional[torch.Tensor]):
        """u -> x. Returns (x_norm, log|det dx/du| in normalised units)."""
        logdet = torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)
        h = u
        for act, cpl in zip(reversed(self.actnorms), reversed(self.couplings)):
            h, ld = cpl.generate(h, ctx)
            logdet = logdet + ld
            h, ld = act.generate(h)
            logdet = logdet + ld
        return h, logdet

    @staticmethod
    def _base_log_prob(u: torch.Tensor) -> torch.Tensor:
        return -0.5 * (u.pow(2).sum(-1) + u.shape[-1] * math.log(2 * math.pi))

    # -- the RL surface ---------------------------------------------------------------
    def log_prob(self, x_raw: torch.Tensor, ctx: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:
        """Exact `log p(x | ctx)` at arbitrary `x`, in raw action units. One pass.

        This is the PPO/GRPO ratio and the KL-to-reference operation. Because the flow is
        invertible and the normalisation is a fixed diagonal affine, it is exact -- there
        is no bound, no sampling, and no decoder to break change-of-variables.
        """
        u, logdet = self._normalising(self.to_normalised(x_raw), ctx)
        return self._base_log_prob(u) + logdet + self.log_unit_det

    def sample(self, ctx: Optional[torch.Tensor] = None, n: Optional[int] = None,
               temperature: float = 1.0,
               generator: Optional[torch.Generator] = None,
               u: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised sample and its log-density, from ONE forward pass.

        Returning both together is deliberate: sampling already computes the
        log-determinant, so a second `log_prob` call would double the cost and invite the
        classic bug where the two disagree by a sign convention.

        Gradients flow through `x` w.r.t. the parameters and the context (the base draw is
        the only stochastic node), so SAC-style pathwise gradients work as-is.

        `temperature != 1` scales the base draw. It is exposed for evaluation sweeps, and
        the returned log-density is still the density of the *model*, not of the tempered
        distribution -- so do not feed tempered samples to an importance ratio.
        """
        if u is None:
            batch = n if n is not None else (1 if ctx is None else ctx.shape[0])
            device = self.unit_scale.device if ctx is None else ctx.device
            dtype = self.unit_scale.dtype if ctx is None else ctx.dtype
            u = torch.empty(batch, self.dim, device=device, dtype=dtype)
            u.normal_(generator=generator)
            if temperature != 1.0:
                u = u * temperature
        x_norm, logdet = self._generating(u, ctx)
        log_prob = self._base_log_prob(u) - logdet + self.log_unit_det
        return self.to_raw(x_norm), log_prob

    def entropy(self, ctx: Optional[torch.Tensor] = None, n_samples: int = 8,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Monte-Carlo `H = -E_x[log p(x)]`, per context row.

        Unbiased, and free while sampling -- no analytic approximation is needed or wanted.
        With no decoder there is no unidentifiable flat region for an entropy bonus to
        inflate for free, which was the pathology `RL_INTERFACE.md` flags for the
        autoencoder design.
        """
        if ctx is None:
            _, lp = self.sample(None, n=n_samples, generator=generator)
            return -lp.mean().unsqueeze(0)
        b = ctx.shape[0]
        rep = ctx.repeat_interleave(n_samples, dim=0)
        _, lp = self.sample(rep, generator=generator)
        return -lp.view(b, n_samples).mean(dim=1)

    @torch.no_grad()
    def mode(self, ctx: Optional[torch.Tensor] = None, n: Optional[int] = None
             ) -> torch.Tensor:
        """The base distribution's mode pushed forward (`eps = 0`).

        Cheap and fully deterministic. It is *not* the mode of the pushforward -- a flow
        has no closed form for that -- which is why `best_of_k` exists alongside it.
        """
        batch = n if n is not None else (1 if ctx is None else ctx.shape[0])
        device = self.unit_scale.device if ctx is None else ctx.device
        dtype = self.unit_scale.dtype if ctx is None else ctx.dtype
        u = torch.zeros(batch, self.dim, device=device, dtype=dtype)
        x, _ = self.sample(ctx, u=u)
        return x

    @torch.no_grad()
    def best_of_k(self, ctx: Optional[torch.Tensor] = None, k: int = 16,
                  seed: int = 0, temperature: float = 1.0) -> torch.Tensor:
        """Highest-log-density of `k` seeded samples. Deterministic given `seed`.

        The evaluation path that actually approximates the mode. It is seeded rather than
        merely stochastic because every closed-loop number becomes irreproducible
        otherwise, and an irreproducible success rate is not a measurement.
        """
        device = self.unit_scale.device if ctx is None else ctx.device
        g = torch.Generator(device=device).manual_seed(int(seed))
        if ctx is None:
            x, lp = self.sample(None, n=k, temperature=temperature, generator=g)
            return x[lp.argmax()].unsqueeze(0)
        b = ctx.shape[0]
        rep = ctx.repeat_interleave(k, dim=0)
        x, lp = self.sample(rep, temperature=temperature, generator=g)
        pick = lp.view(b, k).argmax(dim=1)
        x = x.view(b, k, self.chunk_len, self.n_channels)
        return x[torch.arange(b, device=device), pick]

    # -- diagnostics ------------------------------------------------------------------
    @torch.no_grad()
    def stability_report(self, x_raw: torch.Tensor, ctx: Optional[torch.Tensor] = None
                         ) -> Dict[str, float]:
        """The numbers that say whether the flow is diverging, not whether it is good.

        `saturation` is the fraction of coupling scales pinned at the `s_max` bound. It is
        the early warning: the bound stops a runaway from becoming a NaN, so a flow that
        wants to diverge shows up here as saturation long before anything else moves.
        """
        u, logdet = self._normalising(self.to_normalised(x_raw), ctx)
        sat, n = 0.0, 0
        h = self.to_normalised(x_raw)
        for act, cpl in zip(self.actnorms, self.couplings):
            h, _ = act.normalise(h)
            s, _ = cpl._params(h * cpl.mask, ctx)
            keep = (1.0 - cpl.mask).bool()
            sat += float((s[:, keep].abs() > 0.98 * cpl.s_max).float().sum())
            n += int(s[:, keep].numel())
            h, _ = cpl.normalise(h, ctx)
        return {
            "u_abs_max": float(u.abs().max()),
            "u_std": float(u.std()),
            "logdet_mean": float(logdet.mean()),
            "logdet_abs_max": float(logdet.abs().max()),
            "scale_saturation": sat / max(1, n),
            "actnorm_log_scale_absmax": float(
                max(a.log_scale.abs().max() for a in self.actnorms)),
        }

    # -- checkpointing ----------------------------------------------------------------
    def config(self) -> dict:
        return {
            "context_dim": self.context_dim, "n_layers": self.n_layers,
            "hidden": self.hidden, "depth": self.depth, "s_max": self.s_max,
            "chunk_len": self.chunk_len, "n_channels": self.n_channels,
            "channel_scale": self.unit_scale[: self.n_channels].tolist(),
            "noise_std": self.noise_std_raw[: self.n_channels].tolist(),
            "mask_seed": self.mask_seed,
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "ConditionalFlow":
        return cls(**{k: v for k, v in cfg.items()})


class ContextEncoder(nn.Module):
    """VLM turn vector -> the flow's conditioning vector.

    Separate from the flow so a frozen reference policy can share it during RL without
    duplicating the conditioning trunk (`RL_INTERFACE.md` asks for exactly that), and so
    the flow itself stays usable unconditionally for the marginal-fit control.
    """

    def __init__(self, in_dim: int, out_dim: int = 128, hidden: int = 512,
                 depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers, d = [nn.LayerNorm(in_dim)], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Mish(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def flow_nll(flow: ConditionalFlow, x_raw: torch.Tensor,
             ctx: Optional[torch.Tensor] = None, sigma_mult: float = 1.0,
             generator: Optional[torch.Generator] = None
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row NLL in nats per dimension, plus the dequantised targets used.

    Nats per dimension rather than per chunk so the number can be read against
    `min_nll_per_dim()` without arithmetic, and so it does not silently change meaning if
    the chunk length ever does.
    """
    x = flow.dequantise(x_raw, sigma_mult=sigma_mult, generator=generator)
    return -flow.log_prob(x, ctx) / flow.dim, x
