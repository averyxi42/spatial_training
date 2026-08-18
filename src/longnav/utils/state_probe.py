"""The state probe: distributional distance + value heads on a shared readout hidden.

One trunk, two heads (docs/STOP_HEAD_PLAN.md, consolidation section). Both heads read the
SAME per-turn hidden -- the critic readout token (`readout_offset` from the policy
sandwich, the RL side's `ValueHeadConfig.readout_offset` contract) -- through separate
MLPs. Distance is policy-independent perception (can freeze after SFT); value is
policy-conditional (refreshes with pi during RL).

Both heads are distributional with HL-Gauss targets (cross-entropy against a Gaussian
smeared over the bin support -- bounded, scale-free gradients; see
`vlm_worker.DistributionalValueHead`, whose formulation this generalizes):

* ``LogDistanceHead``: uniform bins in ``log1p(d)`` -- fine near the goal, coarse far,
  with one parameter. Measured against the 20.6M-label v2_25hz sidecar (2026-08-18):
  64 bins over [0, 40 m] give widths 5.5 cm at the goal -> 1.1 m at 20 m, occupancy
  0.15x-3.9x uniform with 3 underfull bins -- acceptably balanced without hand-derived
  quantile edges. The stop rule is ``p_within(logits, r)`` = calibrated P(d <= r),
  thresholdable per benchmark radius (0.4 / 1.0 / 1.6) with no retraining. The scalar
  for baseline features is ``expectation()`` against METER-space centers (never
  exp-of-log-expectation, which carries Jensen bias).
* ``ValueDistHead``: uniform bins on a linear return support, HL-Gauss targets --
  the SFT-side twin of the RL trainer's ``DistributionalValueHead``.

``distance_return_targets`` computes demo returns from a distance series under the RL
reward (clipped geodesic delta) at a given gamma. Returns are PRECOMPUTED into dataset
columns at conversational-format time (the format script already pays a CPU pass), never
derived in the collator: the gamma is stamped alongside so mixed-gamma corpora are
visible instead of silent.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

import torch
import torch.nn as nn

STATE_PROBE_FILE = "state_probe.pt"
STATE_PROBE_CONFIG_FILE = "state_probe_config.json"


# ---------------------------------------------------------------------------
# Return targets from a distance series (pure, no torch needed by callers)
# ---------------------------------------------------------------------------


def distance_return_targets(
    distances: Sequence[Optional[float]],
    gamma: float,
    reward_clip: float = 0.75,
) -> List[float]:
    """Discounted return-to-go under the RL reward, from per-observation distances.

    ``r_k = clip(d_k - d_{k+1}, +-reward_clip)`` -- the training env's reward at
    policy-step cadence, which equals observation cadence for the 2.5 Hz corpora
    (both are 0.4 s). A transition touching a non-finite distance pays 0, mirroring
    the env's ``isfinite`` guard. The final observation's return is 0 (demos end at
    the goal; there is no tail to bootstrap).
    """
    n = len(distances)
    d = [float("nan") if v is None else float(v) for v in distances]
    out = [0.0] * n
    running = 0.0
    for k in range(n - 2, -1, -1):
        if math.isfinite(d[k]) and math.isfinite(d[k + 1]):
            r = max(-reward_clip, min(reward_clip, d[k] - d[k + 1]))
        else:
            r = 0.0
        running = r + gamma * running
        out[k] = running
    return out


# ---------------------------------------------------------------------------
# Distributional scalar heads
# ---------------------------------------------------------------------------


def hl_gauss_targets(u: torch.Tensor, edges: torch.Tensor, sigma: float) -> torch.Tensor:
    """Per-bin mass of N(u, sigma) over ``edges``, support-clamped and renormalized."""
    g = u.float().clamp(float(edges[0]), float(edges[-1]))
    z = (edges.to(g.device) - g.unsqueeze(-1)) / (sigma * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf(z))
    p = cdf[..., 1:] - cdf[..., :-1]
    return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class _DistributionalScalarHead(nn.Module):
    """Categorical head over a monotone transform of a scalar target.

    Subclasses define ``_to_u`` (target -> bin space) and provide ``edges_u``. The
    expectation decodes against ORIGINAL-space bin centers, so no inverse-transform
    bias enters.
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], edges_u: torch.Tensor,
                 centers_orig: torch.Tensor, hl_sigma_ratio: float,
                 dropout: float = 0.0, dtype: Any = torch.float32):
        super().__init__()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        layers: List[nn.Module] = []
        curr = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(curr, h, dtype=dtype), nn.Mish(), nn.Dropout(dropout)]
            curr = h
        layers.append(nn.Linear(curr, len(edges_u) - 1, dtype=dtype))
        self.mlp = nn.Sequential(*layers)
        self.dtype = dtype
        self.register_buffer("edges_u", edges_u.float())
        self.register_buffer("centers_orig", centers_orig.float())
        # sigma in u-space, from the MEAN u-bin width (uniform for both subclasses)
        self.hl_sigma = float(hl_sigma_ratio) * float(
            (edges_u[-1] - edges_u[0]) / (len(edges_u) - 1))

    def _to_u(self, y: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def targets(self, y: torch.Tensor) -> torch.Tensor:
        return hl_gauss_targets(self._to_u(y), self.edges_u, self.hl_sigma)

    def loss(self, logits: torch.Tensor, y: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Masked-mean CE. NaN targets are masked out automatically (sidecar gaps)."""
        finite = torch.isfinite(y)
        m = finite if mask is None else (mask.bool().to(y.device) & finite)
        if not bool(m.any()):
            return logits.sum() * 0.0
        logp = torch.log_softmax(logits.float(), dim=-1)
        ce = -(self.targets(torch.where(finite, y, torch.zeros_like(y))) * logp).sum(-1)
        mf = m.float()
        return (ce * mf).sum() / mf.sum().clamp_min(1.0)

    def expectation(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=-1)
        return probs @ self.centers_orig.to(probs.device)


class LogDistanceHead(_DistributionalScalarHead):
    """Distance-to-goal, categorical over uniform log1p bins."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (1024, 512),
                 n_bins: int = 64, d_max: float = 40.0, hl_sigma_ratio: float = 0.75,
                 dropout: float = 0.0, dtype: Any = torch.float32):
        edges_u = torch.linspace(0.0, math.log1p(d_max), n_bins + 1)
        edges_m = torch.expm1(edges_u)
        centers_m = 0.5 * (edges_m[:-1] + edges_m[1:])
        super().__init__(input_dim, hidden_dims, edges_u, centers_m,
                         hl_sigma_ratio, dropout, dtype)
        self.register_buffer("edges_m", edges_m)
        self.d_max = float(d_max)

    def _to_u(self, y: torch.Tensor) -> torch.Tensor:
        return torch.log1p(y.float().clamp(0.0, self.d_max))

    def p_within(self, logits: torch.Tensor, radius: float) -> torch.Tensor:
        """Calibrated P(d <= radius): full bins below + linear fraction of the
        straddling bin (mass uniform within a bin). THE stop rule."""
        probs = torch.softmax(logits.float(), dim=-1)
        e = self.edges_m.to(probs.device)
        below = (e[1:] <= radius).float()
        inside = ((e[:-1] < radius) & (e[1:] > radius)).float()
        frac = (radius - e[:-1]) / (e[1:] - e[:-1]).clamp_min(1e-9)
        w = below + inside * frac.clamp(0.0, 1.0)
        return probs @ w


class ValueDistHead(_DistributionalScalarHead):
    """Return-to-go, categorical over a uniform linear support (SFT twin of the RL
    trainer's DistributionalValueHead; keep supports consistent when handing over)."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (1024, 512),
                 n_bins: int = 51, v_min: float = -8.0, v_max: float = 24.0,
                 hl_sigma_ratio: float = 0.75, dropout: float = 0.0,
                 dtype: Any = torch.float32):
        edges = torch.linspace(v_min, v_max, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        super().__init__(input_dim, hidden_dims, edges, centers,
                         hl_sigma_ratio, dropout, dtype)

    def _to_u(self, y: torch.Tensor) -> torch.Tensor:
        return y.float()


# ---------------------------------------------------------------------------
# Config + the probe module + checkpoint IO
# ---------------------------------------------------------------------------


@dataclass
class StateProbeConfig:
    """Declared in the derived SFT trainer; every field is checkpoint contract."""

    # Readout position relative to the policy sandwich readout -- MUST match the RL
    # side's ValueHeadConfig.readout_offset when the value head is handed over.
    readout_offset: int = -2
    # Straight-through gradient scale into the backbone hidden (0 = fully detached).
    # This is what makes the backbone proximity/outcome-aware; the whole point of
    # co-training over post-hoc probing (the detached probe was tried and shelved).
    grad_scale: float = 0.1
    distance: Optional[dict] = field(default_factory=lambda: {
        "hidden_dims": [1024, 512], "n_bins": 64, "d_max": 40.0,
        "hl_sigma_ratio": 0.75, "loss_weight": 1.0})
    value: Optional[dict] = field(default_factory=lambda: {
        "hidden_dims": [1024, 512], "n_bins": 51, "v_min": -8.0, "v_max": 24.0,
        "hl_sigma_ratio": 0.75, "loss_weight": 1.0, "gamma": 0.97})

    @classmethod
    def from_dict(cls, d: Any) -> "StateProbeConfig":
        if isinstance(d, cls):
            return d
        return cls(**dict(d))


class StateProbe(nn.Module):
    """Both heads over one readout hidden. Loss = weighted sum of per-head CE."""

    def __init__(self, input_dim: int, cfg: StateProbeConfig, dtype: Any = torch.float32):
        super().__init__()
        self.cfg = cfg
        self.distance_head = None
        self.value_head = None
        if cfg.distance is not None:
            kw = {k: v for k, v in cfg.distance.items() if k != "loss_weight"}
            self.distance_head = LogDistanceHead(input_dim, dtype=dtype, **kw)
        if cfg.value is not None:
            kw = {k: v for k, v in cfg.value.items() if k not in ("loss_weight", "gamma")}
            self.value_head = ValueDistHead(input_dim, dtype=dtype, **kw)

    def losses(self, hidden: torch.Tensor,
               distance_targets: Optional[torch.Tensor] = None,
               return_targets: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None) -> dict:
        g = float(self.cfg.grad_scale)
        h = hidden if g >= 1.0 else (hidden * g + hidden.detach() * (1.0 - g))
        out = {}
        if self.distance_head is not None and distance_targets is not None:
            out["probe/distance_loss"] = self.cfg.distance["loss_weight"] * \
                self.distance_head.loss(self.distance_head(h.to(self.distance_head.dtype)),
                                        distance_targets, mask)
        if self.value_head is not None and return_targets is not None:
            out["probe/value_loss"] = self.cfg.value["loss_weight"] * \
                self.value_head.loss(self.value_head(h.to(self.value_head.dtype)),
                                     return_targets, mask)
        return out


def save_state_probe(out_dir, probe: StateProbe) -> None:
    out = Path(out_dir)
    torch.save(probe.state_dict(), out / STATE_PROBE_FILE)
    cfg = {"readout_offset": probe.cfg.readout_offset, "grad_scale": probe.cfg.grad_scale,
           "distance": probe.cfg.distance, "value": probe.cfg.value}
    (out / STATE_PROBE_CONFIG_FILE).write_text(json.dumps(cfg, indent=2))


def load_state_probe(ckpt_dir, input_dim: int, dtype: Any = torch.float32) -> StateProbe:
    ckpt = Path(ckpt_dir)
    cfg = StateProbeConfig.from_dict(json.loads((ckpt / STATE_PROBE_CONFIG_FILE).read_text()))
    probe = StateProbe(input_dim, cfg, dtype=dtype)
    probe.load_state_dict(torch.load(ckpt / STATE_PROBE_FILE, map_location="cpu"))
    return probe
