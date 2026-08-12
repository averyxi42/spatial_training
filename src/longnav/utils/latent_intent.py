"""The stochastic intent variable `c`, and the SFT-only posterior that gives it a reason.

Design and rationale: `docs/LATENT_RL.md`. This module is the whole architectural delta --
the flow head is untouched, `fm_context_dim` stays 1024, and `FlowActionDecoder`'s
`n_context_tokens * d_model == context_dim` assertion never sees a different number.

TWO SYMBOLS, KEPT APART. `h` is the DETERMINISTIC readout the backbone emits (what a v3
checkpoint calls `c` and hands straight to the decoder). `c` is the LATENT INTENT, a random
variable drawn from a distribution parameterised by `h`. `h` is a function of the
observation, so conditioning on it is legitimate; `c` is drawn, so nothing that produces it
may condition on it. Collapsing the two is how a posterior comes to be written as a function
of the very thing it parameterises.

    h = readout(backbone(o))                     (N, 1024)   deterministic
    mu_p, log_sigma_p = LatentSplit(h)                       deterministic in h
    delta_mu = PosteriorEncoder(sg(h), A)                    SFT only, discarded at RL
    c ~ N(mu_p + delta_mu, diag sigma_p^2)                   the RL action
    A_hat = FlowActionCodec.generate(c)

WHY THE POSTERIOR CONDITIONS ON `h` AND NOT ON THE FULL POOLED STATE. `p` depends on the
observation ONLY through `h`, by construction. A feature of `o` that `h` drops is a feature
`p` cannot express, so a posterior fed the 2048-d pooled state could push `mu_q` along a
direction `p` can never follow; the residual would surface as KL, and the KL is what sets
`sigma_p`, and `sigma_p` IS the exploration distribution at RL time. That reads as "the
outcome is unpredictable here" when the truth is "the prior was never shown the relevant
feature". Aligning the posterior's information set with the prior's is what keeps the KL a
measurement of what the chunk adds. The cost is bounded and measured: the readout MLP
discards 0.0007 R^2 against the pooled state for chunk prediction.

WHY THE RESIDUAL IS NOT EXTRA INFORMATION. `mu_p = W_mu h + b_mu` is a deterministic
function of `h`, which the posterior already conditions on, so `mu_q = mu_p + delta_mu(h, A)`
changes the parameterisation and not the hypothesis class. Its value is COUPLING: without it
`q` and `p` are two independently-parameterised networks chasing each other, and the lag
between them lands in the KL as information that is not there. The residual makes "the chunk
adds nothing here" representable exactly, as `delta_mu = 0`.

THE HAZARD THE RESIDUAL DOES INTRODUCE. Zero-init plus weight decay is a standing pull
toward `delta_mu = 0`, which is `sigma_p -> 0`, which is the degenerate solution -- a
regulariser pointed at the failure mode. `no_decay_parameters()` exists for exactly this;
use it when building the optimiser, and watch `delta_mu_over_sigma` from step 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence, Tuple

import torch
import torch.nn as nn

#: `sample` draws `c ~ N(mu_q_or_p, sigma^2)`; `mean` takes `mu` and is the EXACT v3 path at
#: init, which is what makes the parity assertion a bit-identity rather than a tolerance.
MODES = ("mean", "sample")


class LatentSplit(nn.Module):
    """`h -> (mu, log_sigma)`, initialised so that `mean` mode is the identity.

        W_mu = I,  b_mu = 0        =>  mu = h
        W_sigma = 0, b_sigma = log sigma_0  =>  sigma = sigma_0 everywhere

    So a warm start from a deterministic checkpoint reproduces it exactly in `mean` mode and
    perturbs it by an isotropic `sigma_0` in `sample` mode. `tests/test_latent_split.py`
    asserts the identity as a bit-identity, because that single assertion also covers where
    the split sits, checkpoint loading, and the K-expansion order in the objective.

    OPTIONAL ROTATION. `W_mu = I` leaves `c` in whatever basis the readout MLP happened to
    produce, aligned with nothing, so an axis-aligned `sigma` explores along directions the
    data never occupies. Passing an orthogonal `rotation` R works the split in `R h`'s basis
    and rotates back before the decoder: at `sigma = 0` that is `R^T R h = h`, so parity is
    untouched, while the effective noise in `h`-space becomes `R^T diag(sigma) eps` with
    covariance `R^T diag(sigma^2) R` -- a FULL-COVARIANCE policy at diagonal cost, with
    log-probs still diagonal in the rotated space so no RL math changes. Take R from `h`'s
    PCA basis. Default OFF: adopt it only if the measured spectrum is strongly anisotropic.
    """

    def __init__(self, dim: int = 1024, sigma0: float = 0.02,
                 rotation: Optional[torch.Tensor] = None):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if not (sigma0 > 0.0):
            raise ValueError(f"sigma0 must be positive, got {sigma0}")
        self.dim = int(dim)
        self.sigma0 = float(sigma0)

        self.to_mu = nn.Linear(dim, dim)
        self.to_log_sigma = nn.Linear(dim, dim)
        with torch.no_grad():
            self.to_mu.weight.copy_(torch.eye(dim))
            self.to_mu.bias.zero_()
            self.to_log_sigma.weight.zero_()
            self.to_log_sigma.bias.fill_(math.log(self.sigma0))

        if rotation is None:
            self.register_buffer("rotation", None)
        else:
            R = torch.as_tensor(rotation, dtype=torch.float32)
            if R.shape != (dim, dim):
                raise ValueError(f"rotation must be ({dim}, {dim}), got {tuple(R.shape)}")
            err = float((R @ R.t() - torch.eye(dim)).abs().max())
            if err > 1e-4:
                raise ValueError(
                    f"rotation must be orthogonal -- ||R R^T - I||_inf = {err:.2e}. A "
                    "non-orthogonal basis breaks the parity identity R^T R h = h."
                )
            self.register_buffer("rotation", R)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """`(N, dim) -> (mu, log_sigma)`, both `(N, dim)`, in the ROTATED basis when a
        rotation is set. `mu` is not comparable across rotations; `decode` undoes it."""
        z = h if self.rotation is None else h @ self.rotation.t().to(h.dtype)
        return self.to_mu(z), self.to_log_sigma(z)

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        """Rotated-basis latent -> the basis the decoder consumes. Identity when unrotated."""
        return c if self.rotation is None else c @ self.rotation.to(c.dtype)

    def draw(self, h: torch.Tensor, mode: str = "mean",
             delta_mu: Optional[torch.Tensor] = None,
             generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """One draw per row. Returns the pieces the objective and the diagnostics need.

        `delta_mu` is the posterior's shift; `None` means draw from the PRIOR, which is both
        the rollout path and the RL policy. Returns `c` already in the decoder's basis.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        mu, log_sigma = self(h)
        sigma = log_sigma.exp()
        centre = mu if delta_mu is None else mu + delta_mu
        if mode == "mean":
            c = centre
        else:
            eps = torch.randn(centre.shape, device=centre.device, dtype=centre.dtype,
                              generator=generator)
            c = centre + sigma * eps
        return {"c": self.decode(c), "mu": mu, "log_sigma": log_sigma, "sigma": sigma,
                "centre": centre}

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, sigma0={self.sigma0}, "
                f"rotated={self.rotation is not None}")


class PosteriorEncoder(nn.Module):
    """`q`'s mean shift: `(sg(h), A) -> delta_mu`. **SFT only** -- never loaded at RL or eval.

    Emits `delta_mu` and NOTHING ELSE: run 1 fixes `sigma_q = sigma_p`, which collapses the
    KL to `sum_i delta_mu_i^2 / (2 sigma_p,i^2)`. That is not merely simpler. It leaves
    `sigma_p` with a two-sided gradient -- the flow-matching term pushes it down because
    noise hurts reconstruction, the KL pushes it up because a wider prior makes a given mean
    shift cheaper -- whose balance point is the SCALE OF THE MEAN SHIFT THAT KNOWING THE
    CHUNK INDUCES. That is the quantity the acceptance test wants to read, rather than a free
    parameter. It cannot collapse to zero unless `delta_mu` is zero, which happens only if
    the chunk adds nothing given the observation. Free `sigma_q` is run 2.

    DELIBERATELY UNDER-PARAMETERISED. Capacity here is capacity to over-migrate, so it acts
    as a rate limiter complementing `beta`. But `A` goes in WHOLE AND UNFEATURISED: handing
    it only the terminal displacement and heading would hard-code the mode abstraction this
    design explicitly declines to assume. Give it all `T*3` numbers and measure what fits.
    """

    def __init__(self, dim: int = 1024, n_ticks: int = 20, n_dims: int = 3,
                 width: int = 256):
        super().__init__()
        self.dim, self.n_ticks, self.n_dims = int(dim), int(n_ticks), int(n_dims)
        n_action = self.n_ticks * self.n_dims
        self.action_norm = nn.LayerNorm(n_action)
        self.action_in = nn.Linear(n_action, width)
        self.h_in = nn.Linear(dim, width)
        self.mix = nn.Sequential(
            nn.GELU(), nn.Linear(2 * width, width), nn.GELU(),
        )
        self.out = nn.Linear(width, dim)
        with torch.no_grad():            # KL == 0 exactly at step 0
            self.out.weight.zero_()
            self.out.bias.zero_()

    def forward(self, h: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """`h`: `(N, dim)` -- pass it ALREADY DETACHED. `actions`: `(N, T, 3)` scaled."""
        if actions.shape[1:] != (self.n_ticks, self.n_dims):
            raise ValueError(
                f"actions must be (N, {self.n_ticks}, {self.n_dims}), "
                f"got {tuple(actions.shape)}"
            )
        a = self.action_norm(actions.reshape(actions.shape[0], -1).to(h.dtype))
        return self.out(self.mix(torch.cat([self.action_in(a), self.h_in(h)], dim=-1)))

    def no_decay_parameters(self) -> Iterator[nn.Parameter]:
        """The output layer, which must be EXCLUDED FROM WEIGHT DECAY.

        Zero-init plus decay is a standing pull back toward `delta_mu = 0`, i.e. toward
        `sigma_p -> 0`, i.e. toward the degenerate solution the whole design is trying to
        avoid. It is not an information problem, it is a regulariser aimed the wrong way.
        """
        yield self.out.weight
        yield self.out.bias


def kl_shared_sigma(delta_mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """Per-row `KL(q || p)` in NATS for `sigma_q = sigma_p`: `sum_i dmu_i^2 / (2 sigma_i^2)`.

    The general diagonal-Gaussian KL's `log(sigma_p/sigma_q)` and `sigma_q^2/sigma_p^2` terms
    vanish identically when the scales are shared, leaving a quadratic that is well
    conditioned everywhere -- no `log sigma` in the loss, so no path to a `sigma -> 0`
    singularity through the KL.
    """
    return (delta_mu.pow(2) / (2.0 * (2.0 * log_sigma).exp())).sum(dim=-1)


def latent_diagnostics(pieces: Dict[str, torch.Tensor], delta_mu: torch.Tensor,
                       kl: torch.Tensor, h: torch.Tensor) -> Dict[str, torch.Tensor]:
    """The numbers `docs/LATENT_RL.md` says to log from step 0, as PER-BATCH SUMS.

    Sums rather than means because the trainer accumulates across gradient-accumulation
    steps and ranks and divides by `n_rows` once, which is the only way the logged value is
    a true mean when turn counts differ per batch -- and they differ a lot here (24 to 198
    turns in one probe run).

    `delta_mu_over_sigma` is the one to watch early. It should rise off zero within a few
    hundred steps and settle. Decaying back toward zero while the flow loss is still falling
    means `beta` is too large or weight decay is winning, and the spread gate will fail later
    for a reason that was visible now.

    `h_std_perdim` exists so `sigma_0` can be CALIBRATED from a short probe rather than
    guessed: the design wants it at 1-3% of `h`'s per-dim standard deviation.
    """
    sigma = pieces["sigma"]
    n = float(delta_mu.shape[0])
    per_dim_kl = delta_mu.pow(2) / (2.0 * sigma.pow(2))
    ratio = (delta_mu.abs() / sigma.clamp_min(1e-12)).mean()
    h_std = h.std(dim=0).mean() if h.shape[0] > 1 else h.std()
    return {
        "sum_kl_nats": kl.sum(),
        "sum_sigma": sigma.mean() * n,
        "sum_delta_mu_norm": delta_mu.norm(dim=-1).sum(),
        "sum_delta_mu_over_sigma": ratio * n,
        # Effective dim: dims carrying more than 0.01 nats. An OUTCOME, not a
        # hyperparameter -- and the number that decides whether RL can mask the inactive
        # subspace to make a ratio-based method tractable at 1024 dims.
        "sum_active_dims": (per_dim_kl.mean(dim=0) > 0.01).sum().float() * n,
        "sum_h_std_perdim": h_std * n,
    }


@dataclass
class LatentConfig:
    """What the CLI passes to `TurnFlowActionRegressor.build`.

    `enabled=False` is the whole of v3: no split, no posterior, no KL term, and the
    objective is byte-identical to what it was before this module existed.
    """

    enabled: bool = False
    #: Measured, not chosen: 1-3% of `h`'s per-dim std over the validation split. Flooring
    #: sigma during SFT is deliberately NOT offered -- `sigma_p` is the quantity the
    #: acceptance test reads, and a floor corrupts it. Scale exploration at RL time instead.
    sigma0: float = 0.02
    beta: float = 1.0
    posterior_width: int = 256
    #: Optional orthogonal (dim, dim) PCA basis; see `LatentSplit`. None keeps `h`'s basis.
    rotation: Optional[torch.Tensor] = None

    def describe(self) -> str:
        if not self.enabled:
            return "latent: OFF (deterministic h -> decoder, v3 behaviour)"
        return (f"latent: sigma0={self.sigma0:g} beta={self.beta:g} "
                f"posterior_width={self.posterior_width} "
                f"rotated={self.rotation is not None}")
