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
    stop: Optional[dict] = None      # {"hidden_dims": [256], "pos_weight": 15.0,
                                     #  "radius_m": 1.0, "loss_weight": 1.0}
    value: Optional[dict] = field(default_factory=lambda: {
        "hidden_dims": [1024, 512], "n_bins": 51, "v_min": -8.0, "v_max": 24.0,
        "hl_sigma_ratio": 0.75, "loss_weight": 1.0, "gamma": 0.97})

    @classmethod
    def from_dict(cls, d: Any) -> "StateProbeConfig":
        if isinstance(d, cls):
            return d
        # The config json also carries consumer-facing pins (worker_value_readout_offset,
        # probe_token_id -- see save_state_probe's `extra`); they are not construction
        # arguments, so filter to declared fields instead of exploding on them.
        import dataclasses
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dict(d).items() if k in names})


def binary_stop_log_probs(logits: torch.Tensor) -> torch.Tensor:
    """Score the same bounded Bernoulli used by the STOP behavior sampler."""
    probability = logits.float().sigmoid().double().clamp(1e-6, 1.0 - 1e-6)
    return torch.stack((torch.log1p(-probability), probability.log()), dim=-1).float()


class BinaryStopHead(nn.Module):
    """EPISODE-stop classifier: "is the agent at the goal, should the episode end?"

    NOT the motion-stop head in `stop_head.py`, which asks "should the base stop
    driving?" off the action latent and has been deemed unnecessary (the flow head models
    the whole conditional distribution instead and declares no stop head at all). This one
    reads the PROBE readout hidden and is trained per frame against
    `distance_to_goal <= success_radius` -- a metric label, not the structural
    "final observation of the episode" one, which is wrong for chained-goal PointNav
    (several arrivals per episode) and wrong at deployment (the head must fire on
    ARRIVAL, not when a trajectory happens to end).

    Its output is what terminates an episode under `--stop-head`, replacing the oracle
    `--auto-stop` every result before it relied on.
    """
    """P(within stop radius) from the readout hidden, BCE.

    Deliberately NOT the flow head's stop head: `flow_matching_head` states three times
    that it has no stop head and none should be added, and its context vector is the
    ACTION conditioning, not a state summary. This one reads the same readout hidden the
    distance and value heads read, so it inherits the convention that is already
    parity-checked end to end (offset -2, token-identity pinned).

    Trained with plain `BCEWithLogitsLoss` and a MILD `pos_weight`: the consumer needs a
    calibrated probability to threshold, and AP/AUC are invariant to pos_weight while
    calibration is not -- a weight at the full negative:positive ratio (~38:1 at the
    2.6% positive rate measured on the on-policy corpus) puts the operating point on a
    brittle part of the curve. Fit the threshold on held-out SCENES, not on train.
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (256,),
                 pos_weight: Optional[float] = 15.0, dropout: float = 0.0,
                 dtype: Any = torch.float32):
        super().__init__()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        layers: List[nn.Module] = [nn.LayerNorm(input_dim, dtype=dtype)]
        curr = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(curr, h, dtype=dtype), nn.GELU(), nn.Dropout(dropout)]
            curr = h
        last = nn.Linear(curr, 1, dtype=dtype)
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)   # start at p=0.5, no prior
        layers.append(last)
        self.mlp = nn.Sequential(*layers)
        self.dtype = dtype
        self.pos_weight = None if pos_weight is None else float(pos_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x.to(self.dtype)).squeeze(-1)

    def loss(self, logits: torch.Tensor, y: torch.Tensor,
             mask: Optional[torch.Tensor] = None,
             balanced: bool = False) -> torch.Tensor:
        """Masked-mean BCE. NaN labels are masked, matching the other heads."""
        y = y.to(logits.device)
        finite = torch.isfinite(y)
        m = finite if mask is None else (mask.bool().to(y.device) & finite)
        if not bool(m.any()):
            return logits.sum() * 0.0
        pw = (None if balanced or self.pos_weight is None
              else torch.tensor(self.pos_weight, device=logits.device, dtype=torch.float32))
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits.float(), torch.where(finite, y, torch.zeros_like(y)).float(),
            pos_weight=pw, reduction="none")
        if balanced:
            classes = [m & (y > 0.5), m & (y <= 0.5)]
            return torch.stack([bce[valid].mean() for valid in classes if valid.any()]).mean()
        mf = m.float()
        return (bce * mf).sum() / mf.sum().clamp_min(1.0)

    @staticmethod
    def probability(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits.float())

    def success_probability(self, logits: torch.Tensor, y: torch.Tensor):
        """Exact P(a SAMPLED stop policy ends this episode in-radius) -> (P, arrived).

        Each step is a Bernoulli hazard p_t = sigmoid(s_t) and the episode ends at the
        FIRST stop, so with survival S_t = prod_{i<t}(1 - p_i):

            P(success) = sum_t S_t * p_t * y_t

        a first-passage decomposition -- closed form, O(n), exactly differentiable, no
        sampling at train time. This IS the deployed quantity rather than a surrogate,
        and three things that otherwise need hand-built machinery fall out of it:

        * every step gets gradient, with dP/dp_t = S_t*y_t - P(success after t)/(1 - p_t):
          firing now pays iff in-radius and costs the OPTION VALUE of firing later. The
          trade-off is derived, not imposed by a margin or a temperature.
        * out-of-radius frames following an arrival are penalised for consuming survival
          probability a later arrival needs -- no mask required, and no "first run versus
          all runs" choice to get wrong.
        * a missed first arrival still trains the second, because the gradient simply
          moves to whichever opportunity is left.

        `arrived` is False when no frame is in-radius: P is then identically 0 whatever
        the head does, so those rows are EXCLUDED from the mean rather than dragged
        through it -- their failure belongs to navigation, not to this head.

        Log space throughout; the naive product underflows long before a row's 175 turns.
        """
        y = y.reshape(-1)
        lg = logits.float().reshape(-1)
        finite = torch.isfinite(y)
        yy = torch.where(finite, y, torch.zeros_like(y))
        if not bool((finite & (yy > 0.5)).any()):
            return lg.sum() * 0.0, False
        # unlabelled frames neither fire nor consume survival
        log_p = torch.where(finite, torch.nn.functional.logsigmoid(lg),
                            torch.full_like(lg, -1e30))
        log_1mp = torch.where(finite, torch.nn.functional.logsigmoid(-lg),
                              torch.zeros_like(lg))
        surv = torch.cat([log_1mp.new_zeros(1), torch.cumsum(log_1mp, dim=0)[:-1]])
        return (torch.exp(surv + log_p) * yy).sum(), True

    def firstpass_loss(self, logits: torch.Tensor, y: torch.Tensor,
                       target_eps: float = 0.01):
        """Drive P(success) to (1 - target_eps). -> (loss, arrived).

        The SOFT TARGET is what makes this trainable rather than divergent. Maximising
        P(success) outright has its optimum at a DETERMINISTIC policy -- p -> 1 at the
        best in-radius frame and 0 before it -- i.e. at infinite logits. Measured: the
        unbounded objective's max|logit| goes 8.98 -> 11.29 -> 13.59 at 4k/40k/400k steps
        and is still climbing, while at eps=0.01 it settles at 5.21 by 40k and does not
        move through 400k. That runaway is exactly the score-scale drift already observed
        here (readout norm growing 2.6x monotonically; far-field stop probability
        wandering 0.02 -> 0.31 between adjacent checkpoints).

        eps is not an opaque regulariser: it is the assumed label reliability. Labels come
        from a hard 1.0 m cut on a snapped geodesic, so demanding P = 1 asks the head to
        be more certain than the data is. The penalty is symmetric, so exceeding the
        target is discouraged too -- deliberately, since that is the saturating direction.
        """
        p, arrived = self.success_probability(logits, y)
        if not arrived:
            return p * 0.0, False
        return (p - (1.0 - float(target_eps))) ** 2, True

    @staticmethod
    def threshold_margin_loss(
        logits: torch.Tensor,
        y: torch.Tensor,
        threshold: float,
        positive_margin: float,
        negative_margin: float,
        mode: str = "all",
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Keep logits on the intended side of the deployed hard-stop boundary."""
        if not 0.0 < threshold < 1.0:
            raise ValueError("stop threshold margin threshold must be in (0, 1)")
        if positive_margin < 0.0 or negative_margin < 0.0:
            raise ValueError("stop threshold margins must be non-negative")
        if mode not in {"all", "first_visit"}:
            raise ValueError(f"unknown stop threshold margin mode: {mode}")

        lg = logits.float().reshape(-1)
        yy = y.to(lg.device).float().reshape(-1)
        valid = torch.isfinite(yy)
        if mask is not None:
            valid = valid & mask.to(lg.device).bool().reshape(-1)
        positive = valid & (yy > 0.5)
        selected = valid
        if mode == "first_visit" and bool(positive.any()):
            indices = torch.arange(positive.numel(), device=lg.device)
            first = int(torch.nonzero(positive, as_tuple=False)[0, 0])
            after_first = valid & (indices >= first)
            first_nonpositive = torch.nonzero(
                after_first & ~positive, as_tuple=False
            )
            end = (
                int(first_nonpositive[0, 0])
                if first_nonpositive.numel()
                else positive.numel()
            )
            selected = valid & ((indices < first) | ((indices >= first) & (indices < end)))

        positive = selected & (yy > 0.5)
        negative = selected & ~positive
        decision_logit = math.log(threshold / (1.0 - threshold))
        terms = []
        if bool(positive.any()):
            terms.append(torch.relu(decision_logit + positive_margin - lg[positive]).square().mean())
        if bool(negative.any()):
            terms.append(torch.relu(lg[negative] - (decision_logit - negative_margin)).square().mean())
        if not terms:
            return lg.sum() * 0.0
        return torch.stack(terms).mean()


class StateProbe(nn.Module):
    """Both heads over one readout hidden. Loss = weighted sum of per-head CE."""

    def __init__(self, input_dim: int, cfg: StateProbeConfig, dtype: Any = torch.float32):
        super().__init__()
        self.cfg = cfg
        self.distance_head = None
        self.value_head = None
        self.stop_head = None
        if cfg.distance is not None:
            kw = {k: v for k, v in cfg.distance.items() if k != "loss_weight"}
            self.distance_head = LogDistanceHead(input_dim, dtype=dtype, **kw)
        if cfg.value is not None:
            kw = {k: v for k, v in cfg.value.items() if k not in ("loss_weight", "gamma")}
            self.value_head = ValueDistHead(input_dim, dtype=dtype, **kw)
        self.stop_head = None
        if cfg.stop is not None:
            kw = {k: v for k, v in cfg.stop.items()
                  if k not in ("loss_weight", "radius_m", "firstpass_weight",
                               "target_eps")}
            self.stop_head = BinaryStopHead(input_dim, dtype=dtype, **kw)

    def losses(self, hidden: torch.Tensor,
               distance_targets: Optional[torch.Tensor] = None,
               return_targets: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None,
               stop_targets: Optional[torch.Tensor] = None,
               shadow_stop_actions: Optional[torch.Tensor] = None,
               shadow_stop_rewards: Optional[torch.Tensor] = None,
               shadow_stop_temperature: float = 1.0,
               shadow_stop_weight: float = 1.0,
               firstpass_weight: Optional[float] = None,
               bce_threshold: Optional[float] = None,
               balanced_bce: bool = False,
               bce_weight: float = 1.0,
               threshold_margin_weight: float = 0.0,
               threshold_margin_mode: str = "all",
               threshold_margin_positive: float = 0.0,
               threshold_margin_negative: float = 0.0,
               gradient_diagnostics: bool = False,
               ordered: bool = True) -> dict:
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
        if self.stop_head is not None and stop_targets is not None:
            stop_targets = stop_targets.to(h.device)
            _lg = self.stop_head(h)
            _fp = (float(self.cfg.stop.get("firstpass_weight", 0.0) or 0.0)
                   if firstpass_weight is None else float(firstpass_weight))
            bce_logits = _lg
            if bce_threshold is not None:
                threshold = float(bce_threshold)
                if not 0.0 < threshold < 1.0:
                    raise ValueError("stop BCE threshold must be in (0, 1)")
                import math

                bce_logits = _lg - math.log(threshold / (1.0 - threshold))
            if bce_weight < 0.0:
                raise ValueError("stop BCE weight must be nonnegative")
            _bce = float(bce_weight) * self.cfg.stop.get("loss_weight", 1.0) * \
                self.stop_head.loss(bce_logits, stop_targets, mask, balanced=balanced_bce)
            out["probe/stop_bce_loss"] = _bce
            out["probe/stop_firstpass_loss"] = _lg.sum() * 0.0
            out["probe/stop_threshold_margin_loss"] = _lg.sum() * 0.0
            if _fp > 0.0:
                # ORDER-DEPENDENT: survival is a running product over real time, so a
                # turn permutation makes it meaningless. Shuffled rows fall back to BCE,
                # which still teaches the clock-free part.
                if ordered:
                    _l, _ok = self.stop_head.firstpass_loss(
                        _lg, stop_targets,
                        target_eps=float(self.cfg.stop.get("target_eps", 0.01)))
                    # ALWAYS emit the key; never branch on `_ok`. A row with no arrival
                    # returns a zero still attached to the graph, so every rank composes
                    # the same loss and enters the same collectives. Making a LOSS key
                    # conditional on row CONTENT is what deadlocked DDP at step 0 on
                    # 2026-08-22, and it hung v11 at step 0 the same way before this line.
                    out["probe/stop_firstpass_loss"] = _fp * _l
                else:
                    out["probe/stop_firstpass_loss"] = _lg.sum() * 0.0
            out["probe/stop_loss"] = out["probe/stop_bce_loss"] + \
                out["probe/stop_firstpass_loss"]
            if float(threshold_margin_weight) > 0.0:
                margin_threshold = 0.5 if bce_threshold is None else float(bce_threshold)
                margin_loss = self.stop_head.threshold_margin_loss(
                    _lg,
                    stop_targets,
                    threshold=margin_threshold,
                    positive_margin=float(threshold_margin_positive),
                    negative_margin=float(threshold_margin_negative),
                    mode=threshold_margin_mode,
                    mask=mask,
                )
                out["probe/stop_threshold_margin_loss"] = (
                    float(threshold_margin_weight) * margin_loss
                )
                out["probe/stop_loss"] = (
                    out["probe/stop_loss"] + out["probe/stop_threshold_margin_loss"]
                )
            if shadow_stop_actions is not None and shadow_stop_rewards is not None:
                actions = shadow_stop_actions.to(h.device).float()
                rewards = shadow_stop_rewards.to(h.device).float()
                finite = torch.isfinite(actions) & torch.isfinite(rewards)
                if mask is not None:
                    finite = finite & mask.to(h.device).bool()
                temperature = float(shadow_stop_temperature)
                if temperature <= 0.0:
                    raise ValueError("shadow stop temperature must be positive")
                if bool(finite.any()):
                    log_prob = -torch.nn.functional.binary_cross_entropy_with_logits(
                        _lg.float() / temperature,
                        torch.where(finite, actions, torch.zeros_like(actions)),
                        reduction="none",
                    )
                    shadow_loss = -(log_prob[finite] * rewards[finite]).mean()
                    out["probe/shadow_stop_rl_loss"] = float(shadow_stop_weight) * shadow_loss
                    out["probe/shadow_stop_reward_mean"] = rewards[finite].mean().detach()
                    out["probe/shadow_stop_sample_rate"] = actions[finite].mean().detach()
                    out["probe/stop_loss"] = out["probe/stop_loss"] + out["probe/shadow_stop_rl_loss"]
                else:
                    zero = _lg.sum() * 0.0
                    out["probe/shadow_stop_rl_loss"] = zero
                    out["probe/shadow_stop_reward_mean"] = zero.detach()
                    out["probe/shadow_stop_sample_rate"] = zero.detach()
            if gradient_diagnostics:
                components = {
                    "bce": out["probe/stop_bce_loss"],
                    "firstpass": out["probe/stop_firstpass_loss"],
                    "margin": out["probe/stop_threshold_margin_loss"],
                    "shadow": out.get("probe/shadow_stop_rl_loss", _lg.sum() * 0.0),
                }
                gradients = {}
                for name, component in components.items():
                    gradient = torch.autograd.grad(
                        component, h, retain_graph=True, allow_unused=True
                    )[0]
                    if gradient is not None:
                        gradients[name] = gradient.detach().float().reshape(-1)
                        out[f"probe/grad_{name}_norm"] = gradients[name].norm()
                names = tuple(gradients)
                for left_index, left in enumerate(names):
                    for right in names[left_index + 1:]:
                        denom = gradients[left].norm() * gradients[right].norm()
                        out[f"probe/grad_cos_{left}_{right}"] = (
                            (gradients[left] * gradients[right]).sum()
                            / denom.clamp_min(1e-12)
                        )
        return out

    @torch.no_grad()
    def metrics(self, hidden: torch.Tensor,
                distance_targets: Optional[torch.Tensor] = None,
                return_targets: Optional[torch.Tensor] = None,
                ordered: bool = True,
                stop_targets: Optional[torch.Tensor] = None) -> dict:
        """Interpretable-scale accuracy, as sums + counts so any accumulation
        (grad-accum, DDP, eval loop) averages exactly. Masking matches `loss`:
        non-finite targets (sidecar gaps) contribute nothing. `bias` is signed
        (pred - target): the log-spaced distance bins can skew long-range
        predictions low, and a mean absolute error alone would hide that."""
        out = {}
        if self.distance_head is not None and distance_targets is not None:
            y = distance_targets.float()
            m = torch.isfinite(y)
            if bool(m.any()):
                pred = self.distance_head.expectation(
                    self.distance_head(hidden.to(self.distance_head.dtype))).float()
                err = (pred - y)[m]
                out["probe_dist_abs_err_sum"] = err.abs().sum()
                out["probe_dist_err_sum"] = err.sum()
                out["probe_dist_n"] = m.float().sum()
        if self.stop_head is not None and stop_targets is not None:
            y = stop_targets.float()
            m = torch.isfinite(y)
            if bool(m.any()):
                p = self.stop_head.probability(self.stop_head(hidden))[m]
                yy = y[m]
                out["probe_stop_n"] = m.float().sum()
                out["probe_stop_pos"] = yy.sum()
                # calibration in one number: mean predicted vs actual positive rate.
                # A gap here is the failure mode that ruins a threshold rule even when
                # ranking is good, so it is reported from the first eval, not derived later.
                out["probe_stop_psum"] = p.sum()
                out["probe_stop_tp"] = ((p >= 0.5) & (yy > 0.5)).float().sum()
                out["probe_stop_fp"] = ((p >= 0.5) & (yy <= 0.5)).float().sum()
                out["probe_stop_fn"] = ((p < 0.5) & (yy > 0.5)).float().sum()
            # RAW P(success): the deployment quantity itself, under the sampled stop
            # policy the objective optimises. Averaged over ARRIVING rows only -- a row
            # that never reached the goal scores 0 no matter what the head does, so
            # including it would report navigation failure as head failure.
            if ordered:
                _ps, _ok = self.stop_head.success_probability(
                    self.stop_head(hidden), stop_targets)
                if _ok:
                    out["probe_stop_psuccess_sum"] = _ps.detach()
                    out["probe_stop_psuccess_n"] = torch.ones((), device=_ps.device)
        if self.value_head is not None and return_targets is not None:
            y = return_targets.float()
            m = torch.isfinite(y)
            if bool(m.any()):
                pred = self.value_head.expectation(
                    self.value_head(hidden.to(self.value_head.dtype))).float()
                err = (pred - y)[m]
                out["probe_value_abs_err_sum"] = err.abs().sum()
                out["probe_value_n"] = m.float().sum()
        return out


def save_state_probe(out_dir, probe: StateProbe, extra: Optional[dict] = None) -> None:
    """`extra` carries facts only the TRAINER knows but the RL/parity side needs
    verbatim -- e.g. `worker_value_readout_offset` (the probe position expressed in
    the rollout worker's end-of-turn frame, absorbing shift_left and the template
    content length) and `probe_token_id` (the pinned identity of the readout token).
    Persisting them is what lets a consumer ASSERT the convention instead of
    re-deriving it and being silently off by one."""
    out = Path(out_dir)
    torch.save(probe.state_dict(), out / STATE_PROBE_FILE)
    cfg = {"readout_offset": probe.cfg.readout_offset, "grad_scale": probe.cfg.grad_scale,
           "distance": probe.cfg.distance, "value": probe.cfg.value,
           "stop": probe.cfg.stop}
    if extra:
        cfg.update(extra)
    (out / STATE_PROBE_CONFIG_FILE).write_text(json.dumps(cfg, indent=2))


def load_state_probe(ckpt_dir, input_dim: int, dtype: Any = torch.float32) -> StateProbe:
    ckpt = Path(ckpt_dir)
    cfg = StateProbeConfig.from_dict(json.loads((ckpt / STATE_PROBE_CONFIG_FILE).read_text()))
    probe = StateProbe(input_dim, cfg, dtype=dtype)
    probe.load_state_dict(torch.load(ckpt / STATE_PROBE_FILE, map_location="cpu"))
    return probe
