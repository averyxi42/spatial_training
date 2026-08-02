"""
Discrete-bin coding of an action chunk, for the cross-entropy control head.

This is *deliberately the crude option*. It exists to answer one question -- does any
distributional head stop the policy creeping, or is the flow specifically needed -- and
its two known weaknesses (aliasing from discretization, and a fully factorized model of
the joint distribution over a chunk) are what make it a control rather than a candidate.
Nothing here should grow an autoregressive decoder.

What is coded
-------------
Not the chunk's poses. A chunk is 10 poses *all relative to the same anchor*, so its
coordinates are cumulative and highly correlated -- a bin on `chunk[9].x` spans the whole
episode-scale range and says nothing about whether the robot moved during tick 9. The
quantity with the stop-or-drive structure is the *per-tick body-frame differential*
`(dx, dy, dtheta)`, exactly as `dump/data_diagnostics/analyze_chunk_motion.py` computes
it. So the codec is

    anchor-relative chunk  --decompose-->  10 x (dx, dy, dtheta)  --bin-->  10 x 3 labels

and back. There are 10 differentials, not 9: the anchor is an implicit 11th pose at the
origin and `anchor -> chunk[0]` is a real tick. (`tick_differentials` takes a `stride`
argument to de-overlap consecutive chunks when *estimating a distribution*; for modeling
one chunk we want all 10.)

Why the bin edges are what they are
-----------------------------------
Measured on 675k training ticks (`fit_bins.py` writes the numbers):

  * `dx` and `dy` have a hard atom at exactly 0.0 (34.6% of ticks) surrounded by a cloud
    of float residue out to ~1e-5 m left over from the SE(2) composition of a pose that
    did not actually change. Together that is 62% of ticks for `dx`. This atom is the
    whole phenomenon: the data's stop mass.
  * Away from the atom the data is quantized but not cleanly -- hard spikes at
    +-0.064 m/tick and +-0.08 m/tick for translation, +-0.064 and +-0.08 rad/tick for
    rotation, with genuine continuous spread in between and long thin tails.

Two consequences for the design:

1. **Bin 0 is an exact-zero bin**, `|v| < ZERO_TOL`, and it decodes to bitwise 0.0 -- not
   to a bin centroid that happens to be near zero. A distributional head can only "commit
   to a stop" if choosing the stop label produces a real stop. `ZERO_TOL = 1e-5` is
   deliberately *ten times tighter* than the `EXACT = 1e-4` threshold the band statistics
   in `sweep_analysis.py` measure against, so a good exact-stop score is evidence about
   the model rather than an artifact of aligning the bin edge with the measurement.

2. **The remaining bins are quantile bins over the non-zero values**, per dimension, and
   each decodes to the *mean* of the training values that fall in it (the centroid, which
   minimizes within-bin squared error). A uniform grid would spend most of its bins on the
   near-empty span between the +-0.064 spike and the +-0.08 tail while lumping the entire
   sub-millimetre region -- where the stop-or-drive boundary lives -- into one cell.
   Quantile bins put resolution where the data is, and because the data is spiky the cuts
   naturally crowd around the spikes, which is what keeps the aliasing small.

Edges are pooled across the 10 tick positions within a chunk (one edge set per dimension,
not per tick). The marginal is near-identical at every tick position, and pooling gives
each bin 10x the samples to place its centroid with.

Torch-only (no transformers / peft), so it imports in both the model environment and the
analysis environment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

# Below this, a differential is a stopped robot and nothing else. Chosen from the measured
# atom (see the module docstring) and kept an order of magnitude inside the band
# statistics' EXACT = 1e-4 so the two are not the same threshold wearing two hats.
ZERO_TOL = 1e-5

DIM_NAMES = ("dx", "dy", "dtheta")


# ======================================================================================
# SE(2): chunk poses <-> per-tick body-frame differentials
# ======================================================================================
def wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


def decompose_chunk(chunk: torch.Tensor) -> torch.Tensor:
    """Anchor-relative poses -> per-tick body-frame differentials.

    `chunk`: (..., T, 3) of `(x, y, theta)` all relative to the observation pose.
    Returns (..., T, 3) of `(dx, dy, dtheta)`, the motion *during* each tick expressed in
    the robot's own frame at the start of that tick.

    Identical maths to `analyze_chunk_motion.tick_differentials` with `stride = T`; kept
    here in torch so the training loss and the analysis share one definition.
    """
    zero = torch.zeros_like(chunk[..., :1, :])
    poses = torch.cat([zero, chunk], dim=-2)
    a, b = poses[..., :-1, :], poses[..., 1:, :]
    th = a[..., 2]
    c, s = torch.cos(th), torch.sin(th)
    ddx, ddy = b[..., 0] - a[..., 0], b[..., 1] - a[..., 1]
    return torch.stack(
        [c * ddx + s * ddy, -s * ddx + c * ddy, wrap(b[..., 2] - a[..., 2])], dim=-1
    )


def compose_chunk(diffs: torch.Tensor) -> torch.Tensor:
    """Per-tick body-frame differentials -> anchor-relative poses. Inverse of the above.

    Sequential by construction: tick `t`'s body frame is only known once ticks `< t` have
    been integrated. `T` is 10, so the Python loop is not worth vectorizing away.
    """
    T = diffs.shape[-2]
    x = torch.zeros_like(diffs[..., 0, 0])
    y, th = torch.zeros_like(x), torch.zeros_like(x)
    out = []
    for t in range(T):
        dx, dy, dth = diffs[..., t, 0], diffs[..., t, 1], diffs[..., t, 2]
        c, s = torch.cos(th), torch.sin(th)
        x = x + c * dx - s * dy
        y = y + s * dx + c * dy
        th = wrap(th + dth)
        out.append(torch.stack([x, y, th], dim=-1))
    return torch.stack(out, dim=-2)


# ======================================================================================
# Bin edges
# ======================================================================================
def fit_bin_edges(
    diffs: torch.Tensor,
    n_bins: int,
    zero_tol: float = ZERO_TOL,
) -> Dict[str, torch.Tensor]:
    """Fit one edge set per dimension from observed differentials.

    `diffs`: (N, T, 3) or (N, 3) of per-tick differentials, pooled over tick positions.
    `n_bins`: total labels per dimension, *including* the exact-zero bin at index 0.

    Returns buffers `cuts` (3, n_bins - 2) and `centroids` (3, n_bins). `cuts[d]` are the
    interior boundaries of the non-zero region for dimension `d`, so a value `v` with
    `|v| >= zero_tol` lands in bin `1 + searchsorted(cuts[d], v)`.
    """
    if n_bins < 4:
        raise ValueError("n_bins must be at least 4 (zero bin + a few non-zero bins)")
    flat = diffs.reshape(-1, diffs.shape[-1]).double()
    n_nz = n_bins - 1
    cuts = torch.zeros(flat.shape[-1], n_bins - 2, dtype=torch.float64)
    centroids = torch.zeros(flat.shape[-1], n_bins, dtype=torch.float64)
    for d in range(flat.shape[-1]):
        v = flat[:, d]
        nz = v[v.abs() >= zero_tol]
        if nz.numel() < n_nz:
            raise ValueError(f"dimension {d} has too few non-zero samples to fit bins")
        # Interior quantiles of the NON-ZERO values only. Including the atom would spend
        # most of the cuts inside it, which is precisely the failure mode being avoided.
        q = torch.linspace(0, 1, n_nz + 1, dtype=torch.float64)[1:-1]
        cut = torch.quantile(nz, q)
        # Spiky data produces duplicate quantiles (many samples share one value). Nudge
        # them apart so `searchsorted` still yields distinct, ordered bins -- a duplicated
        # edge would make one bin unreachable and leave its centroid undefined.
        cut = torch.cummax(cut, dim=0).values
        eps = torch.arange(cut.numel(), dtype=torch.float64) * 1e-12
        cut = cut + eps
        cuts[d] = cut
        idx = 1 + torch.searchsorted(cut.contiguous(), nz.contiguous())
        # Centroid = within-bin mean, the squared-error-optimal representative. Bins that
        # ended up empty fall back to their interval midpoint.
        sums = torch.zeros(n_bins, dtype=torch.float64).scatter_add_(0, idx, nz)
        cnts = torch.zeros(n_bins, dtype=torch.float64).scatter_add_(
            0, idx, torch.ones_like(nz)
        )
        mid = torch.cat([cut[:1], 0.5 * (cut[:-1] + cut[1:]), cut[-1:]])
        cen = torch.where(cnts[1:] > 0, sums[1:] / cnts[1:].clamp(min=1), mid)
        centroids[d] = torch.cat([torch.zeros(1, dtype=torch.float64), cen])
    return {"cuts": cuts, "centroids": centroids}


# ======================================================================================
# The codec module
# ======================================================================================
class BinCodec(torch.nn.Module):
    """Chunk <-> per-(tick, dim) bin labels, as a drop-in for `TargetNormalizer`.

    It occupies the `normalizer` slot of `TurnVectorRegressor` on purpose, because that
    slot's contract is exactly "map between the model's output space and the target's
    units" -- `normalize` for the loss, `denormalize` for anything that consumes an
    action. Sitting there means `VectorRolloutPolicy` drives a binned policy with no
    changes at all: it calls `head(...)` then `normalizer.denormalize(...)` and gets a
    chunk in metres and radians, same as for the regression head. The bin edges live in
    buffers, so they are written into and restored from the checkpoint by the existing
    `save_pretrained` / `load_head_state`.

    `decode` controls how a categorical becomes a number:
        argmax  the mode. What the policy executes; the deterministic analogue of the
                regression head's single output.
        sample  a draw from the categorical. Tests the distribution rather than its mode.
        mean    the probability-weighted centroid. Included only to *demonstrate* that
                averaging a distributional head reproduces the regression head's hedging.
        logits  passthrough, for saving raw predictions and decoding offline.
    """

    DECODES = ("argmax", "sample", "mean", "logits")

    def __init__(self, n_dims: int = 3, n_bins: int = 65, zero_tol: float = ZERO_TOL,
                 decode: str = "argmax"):
        super().__init__()
        self.n_dims, self.n_bins = int(n_dims), int(n_bins)
        self.decode = decode
        # Sticky fallback generator for `sample` decode, consulted by `decode_diffs`
        # whenever a caller does not pass one explicitly -- in particular
        # `denormalize()`, which is the path `VectorRolloutPolicy.step()` uses and has
        # no generator parameter of its own. A closed-loop rollout can therefore
        # reseed this once per episode (`codec.generator =
        # torch.Generator(device=...).manual_seed(episode_seed)`) and get
        # reproducible, per-episode-fixed sampling with no change to the rollout code.
        # `None` (the default) means "use the global default RNG", the old behaviour.
        self.generator: Optional[torch.Generator] = None
        self.register_buffer("cuts", torch.zeros(n_dims, n_bins - 2, dtype=torch.float64))
        self.register_buffer("centroids", torch.zeros(n_dims, n_bins, dtype=torch.float64))
        self.register_buffer("zero_tol", torch.tensor(float(zero_tol), dtype=torch.float64))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    # -- fitting -----------------------------------------------------------------------
    @torch.no_grad()
    def fit(self, diffs: torch.Tensor):
        edges = fit_bin_edges(diffs, self.n_bins, float(self.zero_tol))
        self.cuts.copy_(edges["cuts"])
        self.centroids.copy_(edges["centroids"])
        self.fitted.fill_(True)
        return self

    def _require_fitted(self):
        if not bool(self.fitted):
            raise RuntimeError(
                "BinCodec used before fit(); run dump/bin_head_control/fit_bins.py and "
                "pass --bin-edges, or the head would be trained against labels from "
                "all-zero edges"
            )

    # -- differentials <-> labels ------------------------------------------------------
    @torch.no_grad()
    def encode_diffs(self, diffs: torch.Tensor) -> torch.Tensor:
        """(..., 3) differentials -> (..., 3) long labels."""
        self._require_fitted()
        v = diffs.double()
        out = torch.zeros(v.shape, dtype=torch.long, device=v.device)
        for d in range(self.n_dims):
            col = v[..., d].reshape(-1).contiguous()
            idx = 1 + torch.searchsorted(self.cuts[d].contiguous(), col)
            idx = torch.where(col.abs() < self.zero_tol, torch.zeros_like(idx), idx)
            out[..., d] = idx.clamp(0, self.n_bins - 1).reshape(v.shape[:-1])
        return out

    def decode_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """(..., 3) long labels -> (..., 3) differentials, via the bin centroids."""
        self._require_fitted()
        out = torch.zeros(labels.shape, dtype=torch.float64, device=labels.device)
        for d in range(self.n_dims):
            out[..., d] = self.centroids[d][labels[..., d]]
        return out

    def decode_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """(..., 3, n_bins) probabilities -> (..., 3) probability-weighted centroids."""
        self._require_fitted()
        cen = self.centroids.to(probs.device)  # (3, n_bins)
        return (probs.double() * cen).sum(-1)

    # -- the TargetNormalizer interface ------------------------------------------------
    def normalize(self, chunk: torch.Tensor) -> torch.Tensor:
        """(N, T, 3) anchor-relative chunk -> (N, T, 3) long labels."""
        return self.encode_diffs(decompose_chunk(chunk.double()))

    def denormalize(self, logits: torch.Tensor) -> torch.Tensor:
        """(N, T, 3, n_bins) logits -> (N, T, 3) anchor-relative chunk.

        Shape-changing, unlike `TargetNormalizer.denormalize`, which is the price of
        putting the decode here; every caller (loss, rollout, offline analysis) wants a
        chunk in real units out of a model output, and none of them wants logits.
        """
        if self.decode == "logits":
            return logits
        diffs = self.decode_diffs(logits)
        return compose_chunk(diffs)

    def decode_diffs(self, logits: torch.Tensor, decode: Optional[str] = None,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """(N, T, 3, n_bins) logits -> (N, T, 3) per-tick differentials."""
        mode = decode or self.decode
        if mode not in self.DECODES:
            raise ValueError(f"decode must be one of {self.DECODES}, got {mode!r}")
        if mode == "argmax":
            return self.decode_labels(logits.argmax(-1))
        probs = logits.double().softmax(-1)
        if mode == "mean":
            return self.decode_probs(probs)
        flat = probs.reshape(-1, self.n_bins)
        gen = generator if generator is not None else self.generator
        draw = torch.multinomial(flat, 1, generator=gen).reshape(logits.shape[:-1])
        return self.decode_labels(draw)

    # -- persistence -------------------------------------------------------------------
    def to_json(self) -> Dict:
        return {
            "n_dims": self.n_dims,
            "n_bins": self.n_bins,
            "zero_tol": float(self.zero_tol),
            "cuts": self.cuts.tolist(),
            "centroids": self.centroids.tolist(),
        }

    @classmethod
    def from_json(cls, path_or_obj, decode: str = "argmax") -> "BinCodec":
        obj = (json.loads(Path(path_or_obj).read_text())
               if isinstance(path_or_obj, (str, Path)) else path_or_obj)
        codec = cls(obj["n_dims"], obj["n_bins"], obj["zero_tol"], decode=decode)
        codec.cuts.copy_(torch.tensor(obj["cuts"], dtype=torch.float64))
        codec.centroids.copy_(torch.tensor(obj["centroids"], dtype=torch.float64))
        codec.fitted.fill_(True)
        return codec


def roundtrip_report(codec: BinCodec, diffs: torch.Tensor) -> Dict[str, float]:
    """Aliasing cost of the codec alone: encode then decode real data, and measure.

    This is the bin head's analogue of track B's autoencoder reconstruction gate. If the
    round trip already destroys the exact-stop mass or fills the creep band, the codec is
    smearing the data before the model ever sees it and any downstream result is about
    the bins, not about the objective.
    """
    rec = codec.decode_labels(codec.encode_diffs(diffs))
    out: Dict[str, float] = {}
    for d, name in enumerate(DIM_NAMES[: diffs.shape[-1]]):
        err = (rec[..., d] - diffs[..., d].double()).reshape(-1)
        out[f"rmse_{name}"] = float(err.pow(2).mean().sqrt())
        out[f"max_abs_err_{name}"] = float(err.abs().max())
    return out
