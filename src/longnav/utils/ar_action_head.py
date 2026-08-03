"""
An autoregressive-over-ticks action head: `p(a_1, ..., a_T) = prod_t p(a_t | a_<t)`,
where each `a_t` is one tick's WHOLE joint `(dx, dy, dtheta)` decision, vector-quantized
into a learned codebook. The candidate this hedges the flow with, per
`dump/overnight/PLAN.md` track (autoregressive head study) -- see
`dump/autoregressive_head/FINDINGS.md` for the full argument and the design correction
this module implements.

Why NOT one categorical per (tick, dimension) -- the discrete-bin control head's design
(`bin_codec.py`, `bin_head.py`) -- and why not a *sequential* dtheta-then-dx-then-dy
decode within a tick either: a tick's `(dx, dy, dtheta)` is produced by ONE instantaneous
control decision. There is no real causal order between "how much forward motion" and
"how much rotation" the way there genuinely is between tick 4 and tick 5. Either
per-dimension scheme impwoses an artificial ordering on something that is not ordered, and
reproduces the bin head's illegitimate independence assumption at a finer grain (or trades
it for an equally illegitimate, invented sequential dependency). The joint-codebook design
sidesteps this: the vector quantizer sees the true joint `(dx, dy, dtheta)` samples during
fitting (`dump/autoregressive_head/fit_codebook.py`, k-means with a hard-coded exact-zero
atom -- see that file's docstring), so the anti-correlation between translating and turning
is baked into the CODE geometry itself, not asserted by a factorization the model has to
approximate. What genuinely is ordered -- tick 4 before tick 5 -- is exactly what the
autoregression models, chronologically, tick 1 through tick T.

What changes relative to `vector_sft.TurnVectorRegressor` (mirroring `bin_head.py`'s three
changes, plus one more this design needs):

  * `target_shape` is `(context_dim,)`: the shared trunk head (`TurnVectorHead`) is
    repurposed to emit a per-turn CONTEXT VECTOR rather than the full 30-number action
    directly -- the same trick `TurnBinClassifier` uses to size its logit projection, just
    pointed at a fixed-size intermediate instead of the final output.
  * the `normalizer` slot holds an `ARActionCodec` -- a `ChunkVQCodec` (the fitted
    codebook) plus a `CausalActionDecoder` (the tiny transformer) bundled into one module.
    `denormalize(context) -> chunk` is where the substitution happens: instead of one
    function call on logits, it runs the decoder's SEQUENTIAL sampling loop (T forward
    passes, each tick's decode fed back in) and composes the result. This is the only
    place the "autoregressive" in this file's name is spent; everything else is plumbing.
  * `forward` computes cross-entropy over T *joint* K-way categoricals (teacher-forced, one
    parallel forward pass over the true code sequence with a causal mask -- exactly how an
    autoregressive LM is trained) instead of Huber over reals or T x 3 independent
    categoricals.

`vector_sft.py`, `turn_vectors.py`, `vector_rollout.py`, `bin_codec.py` and `bin_head.py`
are all untouched.

--------------------------------------------------------------------------------------
The context pathway (and a known wart in it)
--------------------------------------------------------------------------------------
Naming, because "head" is ambiguous once there are two networks: BACKBONE is the VLM,
READOUT MLP is `TurnVectorHead` (pool -> LayerNorm -> MLP -> context vector), ACTION
DECODER is `CausalActionDecoder` below. The code calls the readout MLP the "trunk head".

    backbone hidden (2048)
      -> readout MLP: [Linear(-> h) -> Mish -> Dropout] per entry of head_hidden_dims,
                      then a final Linear(last_hidden -> context_dim)
      -> context vector (context_dim)
      -> decoder context_proj: Linear(context_dim -> n_context_tokens * d_model)
      -> .view(n_context_tokens, d_model) + start_embed
      -> transformer positions 0 .. n_context_tokens-1

Only two things in that chain constrain what reaches the decoder:

  * `head_hidden_dims`   where the ONLY nonlinearity lives, and how wide it is. This is
                         the real capacity knob.
  * `n_context_tokens * d_model`   a HARD ceiling: a token is d_model wide by definition,
                         so one prefix token caps the context at d_model dimensions no
                         matter how wide the readout MLP is. More tokens is the only way
                         past it.

`context_dim` is NOT a third independent width. The readout MLP's final Linear and
`context_proj` are adjacent linear maps with no activation between them, so their
composition has rank <= min(context_dim, n_context_tokens * d_model) and `context_dim`
functions purely as a rank bound. Set it >= n_context_tokens * d_model (as the current
runs do) and it constrains nothing -- it is then a redundant ~1M-parameter linear factor.

KNOWN WART, kept deliberately rather than silently: `context_proj` is an interface adapter
("accept any context width, project to my d_model") that decouples nothing real, since the
training script must coordinate context_dim / n_context_tokens / d_model in one place
anyway -- which is why `train_ar_action_sft.py` has to WARN when they disagree. The clean
shape is for the readout MLP to emit `n_context_tokens * d_model` directly and for the
decoder to only reshape; then every parameter does work and the constraint is structural
instead of advisory. The one thing that would lose is a LINEAR (activation-free) low-rank
bottleneck, which is a parameter-saving factorization nothing here wants -- a narrow entry
in `head_hidden_dims` gives a better bottleneck, with a nonlinearity at the narrow point.

Historical note: `--head-hidden-dims 128 --context-dim 256` with one token meant the only
nonlinearity ran at width 128 and the decoder could receive 128 dimensions, so the nominal
"256-d context" was never more than 128 dimensions of anything.

--------------------------------------------------------------------------------------
KNOWN GAP: tick 0 is told the robot is at rest, and it is not
--------------------------------------------------------------------------------------
With `use_prefix_pose` on there are THREE input channels per tick:

  1. `token_embed(label_{t-1})`            previous motion, CATEGORICAL  (pre-existing)
  2. state feats 4..6 (dx,dy,dtheta)_prev  previous motion, CONTINUOUS
  3. state feats 0..3 (x, y, cos, sin)     accumulated pose

Channel 3 is the information `use_prefix_pose` was added to supply. Channel 2 is a
continuous restatement of channel 1 -- these features are literally the centroid that
channel 1 looks up, before scaling -- and it is also correlated 0.83/0.77 with channel 3's
xy on real data (identical to it at tick 1 by construction, since one prior differential
IS the accumulated pose there). So the previous motion reaches the model three ways.

Tick 0 receives NONE of the three: its previous-token slot holds the learned `BOS`
placeholder, and `_prefix_state` emits `[0, 0, 1, 0, 0, 0, 0]` -- zero pose AND zero
velocity. It is not one redundant copy that is missing, it is every copy.

MEASURED DEFECTS in the feature layout (validation corpus, 120 episodes):

    feature        std      p99    |max|
    x/XY         1.822    7.004    7.204     <- scale constant undershoots badly
    y/XY         0.538    2.024    7.031
    cos          0.058    1.000    1.000     <- near-constant, ~no information
    sin          0.259    0.642    0.660
    dx_prev/XY   0.319    0.800    0.800
    dy_prev/XY   0.087    0.299    0.794
    dth_prev/TH  0.510    0.801    1.261

  * ONE `POSE_XY_SCALE` normalises two quantities whose ranges differ ~9x: the accumulated
    pose (grows to ~0.7 m over a chunk) and a per-tick differential (~0.05 m). Tuned for
    the latter, it leaves `x/XY` with std 1.8 and a max of 7.2 while `dy_prev/XY` sits at
    std 0.087 -- a ~20x spread in feature scale feeding one un-normalised `Linear`.
  * `cos` is near-degenerate. Accumulated heading inside a chunk never approaches +-pi, so
    cos stays in [0.75, 1.0]. The (cos, sin) pair was chosen to avoid wraparound, but there
    is no wraparound to avoid at this horizon -- raw theta or sin alone carries the same
    information in one feature.
  * there is no normalisation before `pose_proj`, unlike the readout MLP which has its own
    `pre_norm` LayerNorm over the backbone states.

A pose-only variant (`POSE_FEAT_DIM = 4`, dropping feats 4..6) is the minimal form of this
channel and isolates "does accumulated pose help" from "does more previous-motion signal
help". Separate scales for pose and differential, or a LayerNorm on the state vector, would
address the scale defect independently of that choice.

That is wrong about the robot. The chunks overlap 50% (`obs_interval` 0.2 s,
`chunk_duration` 0.4 s), so the executor runs `stride` ticks of chunk i-1 and then
re-observes; at the moment chunk i is predicted the base is MID-MOTION, carrying the
velocity of the last tick it executed. Tick 0 is the only tick denied that.

How much it costs, from the probes in `dump/pose_alignment_probe/`:

  * teacher-forced accuracy is 0.4254 at tick 0 against ~0.73 at every other tick -- a
    ~30 pp gap, and the ONLY structural difference is the missing previous differential
  * lag-1 R^2 of the per-tick differentials is 0.9566, i.e. the previous tick explains 96%
    of the next one's variance. Tick 0 is denied exactly that signal
  * tick 0 is simultaneously the only strongly context-driven tick (47.5% of the gradient
    reaching the context, KL(real||zero-context) 3.33 against <=0.21 everywhere else), and
    free-running decode coasts off whatever it produces

So the weakest-conditioned tick is the one the whole chunk is anchored on.

FUTURE EXPANSION -- the data already contains what is missing:
  * training: the differential of the last executed tick before observation i is
    `decompose_chunk(action_chunks[i-1])[stride - 1]` (stride 5, so index 4). It would ride
    into the collator as a per-turn column and into `_prefix_state` as tick 0's `d_prev`
  * rollout: the policy knows what it just executed, so nothing simulator-side is needed --
    the same value `VectorRolloutPolicy` already emitted on the previous step
  * the first observation of an episode genuinely IS at rest, so keep zeros there -- but
    add an 8th feature, a validity bit, because the current all-zero vector conflates
    "genuinely stationary" with "start of chunk, unknown". Without that bit the model
    cannot tell the two apart, and they imply opposite continuations

OTHER KNOWN REDUNDANCIES AND ODDITIES, collected so they are not rediscovered:

  * `start_embed` duplicates `context_proj.bias` (or the readout MLP's final bias under
    `direct_context`), which is already per-slot after the reshape. Retained for
    checkpoint compatibility; see its definition.
  * `context_proj` is a redundant linear factor whenever `context_dim >=
    n_context_tokens * d_model`. `--direct-context` removes it.
  * under `attn_mode="causal"` the context slots are causally masked among THEMSELVES --
    slot 3 cannot attend to slots 4..7. Harmless (they are carriers, and every tick still
    sees all of them) but arbitrary; `markov` makes them mutually visible.
  * `_prefix_state` recomposes the whole chunk on every one of the T sequential decode
    steps, so the pose channel costs O(T^2) composition work per rollout step. Negligible
    at T=10; it would want a running pose at a larger horizon.
  * `decode(mode="mean")` is not an executable policy and is rejected by the eval backend;
    with `use_prefix_pose` it is a double approximation (its state features are composed
    from greedy codes while its output corresponds to no code sequence).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from longnav.utils.bin_codec import compose_chunk, decompose_chunk
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

# The band edges the existing analysis measures against (sweep_analysis.py / bin_head.py),
# so numbers printed during training sit on the same footing as both existing heads'.
EXACT = 1e-4
CREEP = (0.01, 0.01, 0.02)  # dx, dy, dtheta
DIM_NAMES = ("dx", "dy", "dtheta")

# Rotation-flip deadband, as a SPEED, matching `objectnav_eval.probes` (the closed-loop
# suite) and `dump/autoregressive_head/analyze_ar.py` (the offline one) so the number
# logged during training is the same statistic those two report and cannot drift from
# them. 0.25 * 0.40 rad/s = 0.10 rad/s; scaled by dt to get a per-tick threshold, because
# the corpora run at different tick rates (v1 dt=0.05, v2 dt=0.04) and a fixed per-tick
# radian value would silently mean different speeds on each.
FLIP_DEADBAND_RADPS = 0.25 * 0.40
DEFAULT_DT_NATIVE = 0.04  # v2 (25 Hz); overridden from the dataset by the train script


# ======================================================================================
# The codebook: one joint K-way categorical per tick
# ======================================================================================
class ChunkVQCodec(nn.Module):
    """Per-tick `(dx, dy, dtheta)` <-> a single codebook index. Fit offline by
    `dump/autoregressive_head/fit_codebook.py` (k-means + a hard-coded exact-zero atom at
    index 0); this module only does nearest-centroid encode and centroid-lookup decode.

    Deliberately NOT a drop-in for `TargetNormalizer`'s `normalize`/`denormalize` slot the
    way `BinCodec` is -- decoding this codebook needs the *sequential* decoder loop, which
    only `ARActionCodec` (below) has access to. This class is the lookup table alone.
    """

    DECODES = ("argmax", "sample", "mean", "logits")

    def __init__(self, n_codes: int = 256, n_dims: int = 3, zero_tol: float = 1e-5):
        super().__init__()
        self.n_codes, self.n_dims = int(n_codes), int(n_dims)
        self.register_buffer("centroids", torch.zeros(n_codes, n_dims, dtype=torch.float64))
        self.register_buffer("zero_tol", torch.tensor(float(zero_tol), dtype=torch.float64))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    def _require_fitted(self):
        if not bool(self.fitted):
            raise RuntimeError(
                "ChunkVQCodec used before loading a fitted codebook; run "
                "dump/autoregressive_head/fit_codebook.py and pass --codebook"
            )

    @torch.no_grad()
    def encode_diffs(self, diffs: torch.Tensor) -> torch.Tensor:
        """(..., 3) differentials -> (...,) long codebook indices, nearest centroid.

        Pure nearest-centroid, with no special case for the stop mode. The codebook is
        learned end to end by k-means (`dump/autoregressive_head/fit_codebook.py`) rather
        than having a (0,0,0) atom forced into index 0, so index 0 carries no special
        meaning; an earlier revision overrode stopped ticks to index 0, which would now
        misassign every one of them. The stop code is verified to emerge on its own
        instead -- see that file's `zero_code_report`.
        """
        self._require_fitted()
        v = diffs.double()
        flat = v.reshape(-1, self.n_dims)
        # (n, n_codes) squared distance to every centroid; small enough (n_codes <= a few
        # thousand) that a dense cdist is simpler and fast enough than a KD-tree.
        d2 = torch.cdist(flat, self.centroids) ** 2
        return d2.argmin(dim=-1).reshape(v.shape[:-1])

    def decode_labels(self, idx: torch.Tensor) -> torch.Tensor:
        """(...,) long codebook indices -> (..., 3) differentials."""
        self._require_fitted()
        return self.centroids[idx.reshape(-1)].reshape(*idx.shape, self.n_dims)

    def decode_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """(..., n_codes) probabilities -> (..., 3) probability-weighted centroid."""
        self._require_fitted()
        return torch.einsum("...k,kd->...d", probs.double(), self.centroids)

    def to_json(self) -> Dict:
        return {"n_codes": self.n_codes, "n_dims": self.n_dims,
                "zero_tol": float(self.zero_tol), "centroids": self.centroids.tolist()}

    @classmethod
    def from_json(cls, path_or_obj) -> "ChunkVQCodec":
        """Load a codebook, accepting either on-disk layout.

        Two writers exist and both must load here: this track's
        `dump/autoregressive_head/fit_codebook.py` (`n_clusters` / `centroids` /
        `zero_tol`) and the shared `longnav.utils.action_codebook.ActionCodebook.save`
        (`format` / `n_clusters` / `centroids` / `meta`, and no `zero_tol` at all).
        `ActionCodebook.from_dict` already reads this track's format; this is the
        reciprocal, so a codebook fitted by either component is usable by either consumer.

        `zero_tol` is vestigial and defaults accordingly: it was only ever consulted by the
        joint-zero encode override, which was removed when the codebook stopped reserving
        an index for a forced (0,0,0) atom. It is still stored as a buffer so old
        checkpoints load, but nothing reads it on the encode path.
        """
        obj = (json.loads(Path(path_or_obj).read_text())
               if isinstance(path_or_obj, (str, Path)) else path_or_obj)
        cen = obj.get("centroids")
        if cen is None:
            raise ValueError(f"no 'centroids' in codebook {path_or_obj!r}")
        n_codes = obj.get("n_codes", obj.get("n_clusters", len(cen)))
        if int(n_codes) != len(cen):
            raise ValueError(
                f"codebook declares {n_codes} codes but carries {len(cen)} centroids"
            )
        codec = cls(n_codes=int(n_codes), n_dims=len(cen[0]),
                    zero_tol=float(obj.get("zero_tol", 1e-5)))
        codec.centroids.copy_(torch.tensor(cen, dtype=torch.float64))
        codec.fitted.fill_(True)
        return codec


# ======================================================================================
# The tiny causal transformer: one token per tick, chronological order
# ======================================================================================
class CausalActionDecoder(nn.Module):
    """`p(code_1, ..., code_T | context) = prod_t p(code_t | code_<t>, context)`.

    Genuinely tiny -- a handful of layers over a sequence of length T+1 (T ticks plus one
    context position), never touching the VLM backbone. Standard decoder-only recipe
    (`nn.TransformerEncoder` + a causal mask -- the minGPT trick: self-attention only, no
    cross-attention needed because the context is injected as a token in the sequence
    prefix rather than as a separate cross-attended memory):

      position 0        the per-turn context vector from the VLM trunk, projected to
                         `d_model` and tagged with a learned `start_embed` -- the sequence
                         has no "previous tick" to condition on yet, only image context.
      position t (1..T) `token_embed(code_{t-1})` (the PREVIOUS tick's chosen code, or a
                         dedicated BOS id for t=1, since there is no tick 0) plus
                         `tick_embed(t)` -- the TYPE embedding this design calls for: each
                         position must know WHICH tick it is about to predict, not just
                         count itself off from the previous one. With one token per tick
                         there is no separate channel/dimension to also embed (that was
                         the per-dimension design this one replaces), so `tick_embed`
                         alone carries the full slot identity.

    Causal masking means position t only sees positions <= t, i.e. only ticks < t (plus
    context) -- exactly the chronological factorization the design calls for, and the
    reason teacher-forced training is one parallel forward pass: every position's target
    is available up front, and the mask prevents any position from cheating on its own
    future.
    """

    #: pose/velocity feature layout, documented once so the probes can reproduce it:
    #: [x/S0, y/S0, cos(theta), sin(theta), dx_prev/S1, dy_prev/S1, dtheta_prev/S2]
    POSE_FEAT_DIM = 7

    #: (pose_xy, diff_xy, theta). The first normalises the ACCUMULATED pose, which grows
    #: over a chunk; the second a single tick's differential. They must be separate --
    #: their ranges differ ~9x -- and pose_xy/diff_xy must each apply to BOTH x and y, or
    #: the scaling would be anisotropic and distort the geometry the decoder reasons over.
    #:
    #: LEGACY is the original single-constant layout. It is the CONSTRUCTOR DEFAULT so that
    #: every checkpoint written before this split -- including run_v3_ctx8_pose, which is
    #: training as this is written -- reloads with exactly the feature scaling it was
    #: trained under. Changing the default would silently mis-scale those inputs at eval.
    POSE_SCALES_LEGACY = (0.1, 0.1, 0.1)
    #: Calibrated so each feature lands near unit variance on the v2_25hz corpus:
    #: measured stds were x 0.182 m, y 0.054 m, dx_prev 0.032 m, dy_prev 0.009 m,
    #: dtheta_prev 0.051 rad. New runs should pass this.
    POSE_SCALES_CALIBRATED = (0.15, 0.03, 0.05)

    def __init__(self, context_dim: int, n_codes: int, n_ticks: int, d_model: int = 128,
                 n_layers: int = 4, n_heads: int = 4, dim_ff: int = 512, dropout: float = 0.1,
                 n_context_tokens: int = 1, use_prefix_pose: bool = False,
                 attn_mode: str = "causal", direct_context: bool = False,
                 pose_scales: Optional[Sequence[float]] = None):
        super().__init__()
        if attn_mode not in ("causal", "markov"):
            raise ValueError(f"attn_mode must be causal|markov, got {attn_mode!r}")
        self.n_codes, self.n_ticks, self.d_model = n_codes, n_ticks, d_model
        self.n_context_tokens = int(n_context_tokens)
        self.use_prefix_pose = bool(use_prefix_pose)
        self.attn_mode = attn_mode
        self.direct_context = bool(direct_context)
        self.context_dim = int(context_dim)
        scales = tuple(float(x) for x in (pose_scales or self.POSE_SCALES_LEGACY))
        if len(scales) != 3 or any(x <= 0 for x in scales):
            raise ValueError(
                f"pose_scales must be 3 positive floats (pose_xy, diff_xy, theta), "
                f"got {pose_scales!r}"
            )
        self.pose_scales = scales
        self.BOS = n_codes  # one extra id: "no previous tick" (used only at position 1)

        if self.direct_context:
            # The readout MLP emits token space directly; the decoder only reshapes. This
            # is the clean separation: the redundant linear factor is gone, and the width
            # agreement is a structural requirement rather than the advisory warning the
            # projecting path needs. See the module docstring's context-pathway section.
            want = self.n_context_tokens * d_model
            if self.context_dim != want:
                raise ValueError(
                    f"direct_context requires context_dim == n_context_tokens * d_model "
                    f"({self.n_context_tokens} * {d_model} = {want}), got {context_dim}"
                )
            self.context_proj = None
        else:
            # Legacy path: an interface adapter accepting any context width. Kept as the
            # default so every checkpoint written before `direct_context` still loads.
            self.context_proj = nn.Linear(context_dim, self.n_context_tokens * d_model)
        # A learned per-slot bias on the context tokens. Honest note: this is REDUNDANT
        # with `context_proj.bias`, which already has n_ctx * d_model entries and is
        # therefore per-slot once reshaped -- (Wx + b).view(C,d) + s is the same function
        # as (Wx).view(C,d) + (b.view(C,d) + s). The slots are distinguishable without it
        # anyway, since each reads its own block of rows of W. It survives because it
        # predates the multi-token split and dropping it would break every existing
        # checkpoint; it costs C*d parameters and buys initialization (zeros, so the
        # effective slot bias starts as the Linear's bias alone), not expressive power.
        # It WOULD be load-bearing in a variant that shared one projection across slots.
        self.start_embed = nn.Parameter(torch.zeros(self.n_context_tokens, d_model))
        # Discrete lookup rather than a projection of the centroid: the table can place
        # geometrically adjacent codes arbitrarily far apart, so nothing forces near-zero
        # motions to share an embedding. (A weak argument behaviourally -- a stop and a
        # 1 mm tick are near-identical to act on -- but the table is standard, costs
        # 131k parameters, and removes the question.) Row `n_codes` is BOS.
        self.token_embed = nn.Embedding(n_codes + 1, d_model)
        # The ONLY positional signal for tick positions -- there is no sinusoidal encoding
        # anywhere in this decoder. Also carries "how much of the chunk is left".
        self.tick_embed = nn.Embedding(n_ticks, d_model)
        self.pose_proj = (nn.Linear(self.POSE_FEAT_DIM, d_model)
                          if self.use_prefix_pose else None)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_ff, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, n_codes)
        # kept for checkpoint round-tripping (TurnARActionClassifier.save_pretrained)
        self._init_kwargs = dict(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                                 dim_ff=dim_ff, dropout=dropout,
                                 n_context_tokens=self.n_context_tokens,
                                 use_prefix_pose=self.use_prefix_pose,
                                 attn_mode=self.attn_mode,
                                 direct_context=self.direct_context,
                                 pose_scales=list(self.pose_scales))
        self._register_load_state_dict_pre_hook(self._upgrade_start_embed, with_module=False)

    @staticmethod
    def _upgrade_start_embed(state_dict, prefix, *args, **kwargs):
        """Checkpoints written before multi-token context stored `start_embed` as
        `(d_model,)`; it is now `(n_context_tokens, d_model)`. Unsqueeze on load so the
        existing run's checkpoints (and the offline tooling that reads them) keep working
        without a migration script."""
        key = prefix + "start_embed"
        if key in state_dict and state_dict[key].dim() == 1:
            state_dict[key] = state_dict[key].unsqueeze(0)

    def to_config(self) -> Dict:
        return dict(self._init_kwargs)

    def _tick_embeds(self, device) -> torch.Tensor:
        return self.tick_embed(torch.arange(self.n_ticks, device=device))  # (T, d_model)

    def _attn_mask(self, device, dtype) -> Tuple[torch.Tensor, bool]:
        """Returns (additive float mask over C+T positions, is_causal_hint).

        `causal`  the original lower-triangular mask. Tick t sees every context slot and
                  every earlier tick.

        `markov`  tick t sees every context slot and ITSELF, and no other tick. Only a
                  sensible TRAINING config together with `use_prefix_pose`, because that is
                  what makes each tick's own input a sufficient state (accumulated pose +
                  previous differential + tick index): with the state supplied, the token
                  history is redundant, and
                  masking it turns the decoder into a state-conditioned policy unrolled
                  over ticks rather than a sequence model over codes. That makes
                  history-based heuristics structurally impossible instead of merely
                  unattractive, so it is a direct test of whether the context carries the
                  chunk's global shape. Context slots stay mutually visible -- they are
                  carriers, and letting them mix costs nothing.

                  The CONSTRUCTOR deliberately permits `markov` without `use_prefix_pose`:
                  that combination is the ablation which isolates the attention path (with
                  the state channel off, severing attention must sever ALL cross-tick
                  information), and `tests/test_ar_head_variants.py` relies on it. The
                  guard that matters is in `train_ar_action_sft.py`, which refuses to
                  LAUNCH A RUN in that configuration.

        NOTE: `is_causal=True` is only a HINT to the fast path and permits it to apply pure
        causal masking, which would silently discard a custom mask. It is therefore
        returned as False for any non-triangular mode.
        """
        C, T = self.n_context_tokens, self.n_ticks
        L = C + T
        if self.attn_mode == "causal":
            return nn.Transformer.generate_square_subsequent_mask(L, device=device,
                                                                  dtype=dtype), True
        allow = torch.zeros(L, L, dtype=torch.bool, device=device)
        allow[:, :C] = True                       # everyone may read every context slot
        idx = torch.arange(C, L, device=device)
        allow[idx, idx] = True                    # each tick may read itself
        allow[:C, :C] = True                      # context slots see each other
        mask = torch.zeros(L, L, dtype=dtype, device=device)
        return mask.masked_fill(~allow, float("-inf")), False

    def _prefix_state(self, labels: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """Pose/velocity features describing the state BEFORE each tick. (N, T, 7).

        Position `t` gets the pose accumulated over ticks `< t` and the differential of
        tick `t-1`; both are zero at `t == 0` -- see the module docstring's "KNOWN GAP"
        section, because that zero is a claim the robot is at rest and it is false under a
        receding horizon. This adds NO information -- it is a
        deterministic function of the same prefix the model already conditions on -- it
        only removes a computation the architecture is bad at: composing SE(2) is a
        nonlinear recurrence (products of rotation matrices), not the weighted sum
        attention natively computes.

        Placeholder safety, the same invariant `decode` relies on for tokens: during
        sequential decode positions `>= t` are zero-filled, so `poses[:, >= t]` is
        garbage -- but position `t`'s features depend only on `labels[:, :t]`, which are
        decided. Garbage at later positions is never read for the logits actually used.
        """
        N, T = labels.shape
        diffs = centroids.to(labels.device).index_select(0, labels.reshape(-1))
        diffs = diffs.reshape(N, T, 3).float()
        poses = compose_chunk(diffs.double()).float()            # pose AFTER each tick
        z = poses.new_zeros(N, 1, 3)
        p_prev = torch.cat([z, poses[:, :-1, :]], dim=1)         # pose BEFORE tick t
        d_prev = torch.cat([z, diffs[:, :-1, :]], dim=1)         # differential of tick t-1
        s_pose, s_diff, s_th = self.pose_scales
        return torch.stack([
            p_prev[..., 0] / s_pose, p_prev[..., 1] / s_pose,
            torch.cos(p_prev[..., 2]), torch.sin(p_prev[..., 2]),
            d_prev[..., 0] / s_diff, d_prev[..., 1] / s_diff, d_prev[..., 2] / s_th,
        ], dim=-1)

    def forward(self, context: torch.Tensor, labels: torch.Tensor,
                centroids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Teacher-forced. `context`: (N, context_dim) float. `labels`: (N, T) long, the
        TRUE code at every tick. `centroids`: (K, 3), required only when the decoder was
        built with `use_prefix_pose` (it turns codes into the differentials the pose
        features are composed from). Returns (N, T, n_codes) logits, position t predicting
        `labels[:, t]`."""
        N, T = labels.shape
        device = context.device
        C = self.n_context_tokens
        proj = context if self.direct_context else self.context_proj(context)
        ctx_tok = proj.view(N, C, self.d_model) + self.start_embed

        bos = torch.full((N, 1), self.BOS, dtype=torch.long, device=device)
        prev = torch.cat([bos, labels[:, :-1]], dim=1)               # (N, T)
        tok_e = self.token_embed(prev)                               # (N, T, d)
        slot_e = self._tick_embeds(device).unsqueeze(0)               # (1, T, d)
        tick_x = tok_e + slot_e
        if self.use_prefix_pose:
            if centroids is None:
                raise ValueError(
                    "use_prefix_pose=True requires `centroids`; pass the codec's centroid "
                    "table (ARActionCodec does this for you)"
                )
            tick_x = tick_x + self.pose_proj(
                self._prefix_state(labels, centroids).to(tick_x.dtype)
            )
        x = torch.cat([ctx_tok, tick_x], dim=1)                       # (N, C+T, d)

        mask, is_causal = self._attn_mask(device, x.dtype)
        h = self.blocks(x, mask=mask, is_causal=is_causal)
        return self.out_proj(self.ln_f(h[:, C:, :]))                  # (N, T, n_codes)

    @torch.no_grad()
    def decode(self, context: torch.Tensor, mode: str = "sample",
              generator: Optional[torch.Generator] = None,
              temperature: float = 1.0,
              centroids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequential decode, T forward passes. Correct WITHOUT a KV cache: at step `t`
        the input tensor is `labels` with positions `>= t` still zero-filled placeholders,
        but the causal mask means position `t`'s output depends only on positions `<= t`
        (i.e. only on already-decided ticks), so those placeholders can never leak into
        the logits actually used. Recomputing the full (tiny) sequence each step is
        `O(T^2)` in attention work, trivial at `T ~ 10` and simpler than incremental
        caching -- exactly the "genuinely tiny" brief this component is held to; a real
        deployment-scale version would want a cache, this prototype does not need one to
        be correct or fast enough to evaluate.

        The same placeholder argument covers the pose/velocity channel when
        `use_prefix_pose` is on: position `t`'s features are composed from `labels[:, :t]`
        only, so they are correct at the step they are read, and the garbage composed at
        later positions is discarded by the mask. `test_ar_head_variants.py` asserts this
        rather than leaving it to the argument.
        """
        if mode not in ("argmax", "sample", "mean"):
            raise ValueError(f"decode mode must be argmax/sample/mean, got {mode!r}")
        N, T = context.shape[0], self.n_ticks
        device = context.device
        labels = torch.zeros(N, T, dtype=torch.long, device=device)
        logits_all = torch.zeros(N, T, self.n_codes, device=device, dtype=torch.float32)
        for t in range(T):
            logits = self.forward(context, labels, centroids=centroids)   # (N,T,n_codes)
            lt = logits[:, t, :]
            logits_all[:, t, :] = lt
            if mode == "argmax":
                tok = lt.argmax(-1)
            elif mode == "sample":
                probs = (lt.float() / max(temperature, 1e-6)).softmax(-1)
                tok = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            else:  # mean has no discrete token to feed back; see ARActionCodec.denormalize
                raise ValueError("mode='mean' is not sequentially decodable (no single "
                                 "token to condition future ticks on); use "
                                 "ARActionCodec.denormalize(..., decode='mean') instead, "
                                 "which computes it from a single teacher-forced-style "
                                 "pass for reporting only, not as an executable policy")
            labels[:, t] = tok
        return labels, logits_all


# ======================================================================================
# The normalizer-slot adapter: codebook + decoder, one module, one `denormalize` call
# ======================================================================================
class ARActionCodec(nn.Module):
    """Occupies the `normalizer` slot of `TurnVectorRegressor`'s contract ("convert
    between model space and target units") exactly the way `BinCodec` does for the
    (non-autoregressive) bin head -- so `save_pretrained`/`load_head_state` and
    `VectorRolloutPolicy.step()` work with NO changes to `vector_sft.py` or
    `vector_rollout.py`. The difference from `BinCodec` is what `denormalize` has to do:
    a plain codec maps logits -> chunk in one function call; this one decodes T ticks
    SEQUENTIALLY (feeding each chosen code back in) before composing the chunk. That
    sequential loop is the entire point of this head, so it lives here, one call deep from
    `VectorRolloutPolicy.step()`, rather than upstream in the rollout code.
    """

    DECODES = ("argmax", "sample", "mean", "context")

    def __init__(self, codec: ChunkVQCodec, decoder: CausalActionDecoder):
        super().__init__()
        self.codec = codec
        self.decoder = decoder
        self.decode = "sample"
        self.temperature = 1.0
        self.generator: Optional[torch.Generator] = None

    def normalize(self, chunk: torch.Tensor) -> torch.Tensor:
        """(..., T, 3) anchor-relative chunk -> (..., T) long per-tick code labels."""
        return self.codec.encode_diffs(decompose_chunk(chunk.double()))

    def denormalize(self, context: torch.Tensor) -> torch.Tensor:
        """(N, context_dim) -> (N, T, 3) anchor-relative chunk (or, with `decode ==
        "context"`, the untouched context vector -- a passthrough for
        `dump/autoregressive_head/predict_ar.py`, exactly the role `BinCodec`'s
        `decode = "logits"` plays for the bin head: save the cheap-to-decode-later
        quantity once per real VLM forward pass, and run every decode rule -- including
        several independent `sample` draws -- offline against it with no further GPU
        backbone work, since the decoder alone is tiny enough to run anywhere torch runs.

        `mean` is the one decode rule that is not itself an executable sequential policy
        (there is no single token to condition the next tick on when every code is used
        in proportion to its probability) -- kept anyway, exactly as the bin head kept it,
        because it is the control-within-the-control: if creeping comes back under mean
        decode from this SAME trained network, that is the clean re-confirmation that
        averaging is the failure mode, independent of architecture. It is computed as the
        probability-weighted centroid at each tick GIVEN THE GREEDY-DECODED PREFIX (the
        only prefix available without a token to feed forward), which is an approximation
        for reporting only, not a deployable rule.
        """
        if self.decode not in self.DECODES:
            raise ValueError(f"decode must be one of {self.DECODES}, got {self.decode!r}")
        if self.decode == "context":
            return context
        cen = self.codec.centroids
        if self.decode == "mean":
            tokens, _ = self.decoder.decode(context, mode="argmax", centroids=cen)
            logits = self.decoder.forward(context, tokens, centroids=cen)
            probs = logits.float().softmax(-1)
            diffs = self.codec.decode_probs(probs)
        else:
            tokens, _ = self.decoder.decode(
                context, mode=self.decode, generator=self.generator,
                temperature=self.temperature, centroids=cen,
            )
            diffs = self.codec.decode_labels(tokens)
        return compose_chunk(diffs)


# ======================================================================================
# The model
# ======================================================================================
class TurnARActionClassifier(TurnVectorRegressor):
    """`TurnVectorRegressor` whose head emits a context vector consumed by a
    `CausalActionDecoder` over per-tick VQ codes, instead of a direct regression or a
    per-(tick,dim) categorical output."""

    #: swapped by `ar_action_head_v2.TurnARActionClassifierV2`; see `build`.
    DECODER_CLS = CausalActionDecoder

    @classmethod
    def build(
        cls,
        model_cfg: ModelConfig,
        loss_cfg: LossConfig,
        lora: Optional[LoraSpec],
        n_ticks: int,
        processor,
        n_codes: int = 256,
        context_dim: int = 256,
        decoder_kwargs: Optional[Dict] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "TurnARActionClassifier":
        loss_cfg = LossConfig(**{**loss_cfg.__dict__, "normalize_targets": False})
        target_shape = (context_dim,)   # sizes the trunk head's projection only
        model = super().build(model_cfg, loss_cfg, lora, target_shape, processor, dtype)
        codec = ChunkVQCodec(n_codes=n_codes, n_dims=3)
        # Indirection, not a behaviour change: `ar_action_head_v2` subclasses this and
        # swaps in its own decoder by overriding DECODER_CLS, so `from_pretrained` (which
        # routes through here) rebuilds the right one without a forked loader.
        decoder = cls.DECODER_CLS(context_dim=context_dim, n_codes=n_codes,
                                  n_ticks=n_ticks, **(decoder_kwargs or {}))
        model.normalizer = ARActionCodec(codec, decoder)
        model.n_ticks = int(n_ticks)
        return model

    @property
    def codec(self) -> ChunkVQCodec:
        return self.normalizer.codec

    @property
    def decoder(self) -> CausalActionDecoder:
        return self.normalizer.decoder

    @property
    def n_codes(self) -> int:
        return self.normalizer.codec.n_codes

    # -- the objective -----------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_turns: Optional[torch.Tensor] = None,
        num_items_in_batch: Optional[Union[int, torch.Tensor]] = None,
        **multimodal,
    ) -> Dict[str, torch.Tensor]:
        multimodal.pop("labels", None)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
            **multimodal,
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

        targets = targets.to(context.device)
        gt_diffs = decompose_chunk(targets.double())            # (N, T, 3)
        labels = self.codec.encode_diffs(gt_diffs)               # (N, T) long

        logits = self.decoder(context.float(), labels,
                              centroids=self.codec.centroids)   # (N, T, n_codes)
        ce = F.cross_entropy(
            logits.reshape(-1, self.n_codes), labels.reshape(-1), reduction="none"
        ).view(-1, self.n_ticks)
        per_turn = ce.mean(dim=1)
        total = per_turn.sum()
        denom = num_items_in_batch if num_items_in_batch is not None else per_turn.numel()
        denom = torch.as_tensor(denom, dtype=total.dtype, device=total.device).clamp(min=1)
        loss = total / denom

        with torch.no_grad():
            metrics = self._band_metrics(logits.detach(), labels, gt_diffs)
            if not self.training:
                metrics.update(
                    self._free_running_metrics(context.float(), labels, gt_diffs, targets)
                )
            metrics["loss_sum"] = total.detach()
            metrics["n_turns"] = torch.tensor(logits.shape[0], device=logits.device)
            metrics["n_tokens"] = torch.tensor(
                outputs["last_hidden_state"].shape[1], device=logits.device
            )
            metrics["n_dense_tokens"] = torch.tensor(input_ids.shape[1], device=logits.device)
            metrics["n_steps"] = torch.tensor(1, device=logits.device)
        return {"loss": loss, **metrics}

    @torch.no_grad()
    def _band_metrics(self, logits, labels, gt_diffs) -> Dict[str, torch.Tensor]:
        """Exact-stop / creep-band mass on forward (dx) and rotation (dtheta), plus
        TEACHER-FORCED (argmax-per-position, given the true prefix) accuracy -- the
        training-time number to watch, analogous to the bin head's per-step accuracy.
        This is NOT the same as sequential greedy/sample decode accuracy; that gap is
        exactly the exposure-bias check (`dump/autoregressive_head/predict_ar.py` /
        FINDINGS.md compute both and compare).
        """
        dev = logits.device
        pred = logits.argmax(-1)                                # (N, T) teacher-forced
        pred_diffs = self.codec.decode_labels(pred)              # (N, T, 3)

        used = [0, 1, 2]  # dx, dy, dtheta -- the full table, as bin_head/flow_head report
        D = len(used)
        creep = torch.tensor([CREEP[d] for d in used], dtype=torch.float64, device=dev)

        def bands(v):
            a = v.abs()
            stop = (a < EXACT).reshape(-1, D).double().sum(0)
            crp = ((a >= EXACT) & (a < creep)).reshape(-1, D).double().sum(0)
            return stop, crp

        pred_sel, gt_sel = pred_diffs[..., used], gt_diffs[..., used]
        stop_p, creep_p = bands(pred_sel)
        stop_g, creep_g = bands(gt_sel)
        err = (pred_sel - gt_sel).reshape(-1, D)

        # Joint per-tick token correctness/top-5, reported once and duplicated across ALL
        # "dims" for interface symmetry with the bin head's table -- there is only one
        # categorical draw per tick, so every channel's correctness rises and falls
        # together by construction (documented, not a bug: see the module docstring).
        correct = (pred == labels).double().mean() * gt_diffs.shape[0] * gt_diffs.shape[1]
        topk = (logits.topk(min(5, logits.shape[-1]), dim=-1).indices
                == labels.unsqueeze(-1)).any(-1).double().sum()
        rows = torch.tensor(gt_diffs.shape[0] * gt_diffs.shape[1], device=dev)

        return {
            "sum_sq_err": err.pow(2).sum(0).float(),
            "sum_abs_err": err.abs().sum(0).float(),
            "n_rows": rows,
            "sum_correct": correct.repeat(D).float(),
            "sum_topk": topk.repeat(D).float(),
            "sum_stop_pred": stop_p.float(),
            "sum_stop_gt": stop_g.float(),
            "sum_creep_pred": creep_p.float(),
            "sum_creep_gt": creep_g.float(),
        }

    @torch.no_grad()
    def _free_running_metrics(self, context, labels, gt_diffs, targets) -> Dict[str, torch.Tensor]:
        """GREEDY FREE-RUNNING decode: the deploying path, with no true prefix anywhere.

        This is the metric the teacher-forced table cannot substitute for. Zeroing the
        context costs teacher-forced accuracy ~5pp but costs free-running accuracy ~25pp,
        because teacher forcing lets the decoder predict tick t from the TRUE tick t-1 and
        never have to read the scene. Only this number moves when conditioning improves,
        so any change aimed at context reliance has to be judged here.

        Eval-only: it costs T sequential decoder passes. Trivial next to the VLM backbone
        but not free, and it is not needed every training step.
        """
        dev = context.device
        tokens, _ = self.decoder.decode(context, mode="argmax",
                                        centroids=self.codec.centroids)  # (N,T)
        diffs = self.codec.decode_labels(tokens)                    # (N, T, 3)
        chunk = compose_chunk(diffs)                                # (N, T, 3)

        rows = torch.tensor(labels.shape[0] * labels.shape[1], device=dev)
        correct = (tokens == labels).double().sum()

        # Composed-pose error against the true chunk, in the target's own units. Composed
        # (not per-tick) because that is what the controller actually tracks.
        err = (chunk - targets.double()).reshape(-1, 3)

        # Rotation flips: adjacent WITHIN-CHUNK tick pairs where both ticks rotate past
        # the deadband and the commanded direction reverses. Same definition as the
        # offline/closed-loop probes; see FLIP_DEADBAND_RADPS.
        band = FLIP_DEADBAND_RADPS * float(getattr(self, "dt_native", DEFAULT_DT_NATIVE))
        dth = diffs[..., 2]
        active = dth.abs() > band
        both = active[:, :-1] & active[:, 1:]
        flips = ((torch.sign(dth[:, :-1]) * torch.sign(dth[:, 1:])) < 0) & both

        return {
            "sum_free_correct": correct.repeat(3).float(),
            "sum_free_sq_err": err.pow(2).sum(0).float(),
            "sum_free_abs_err": err.abs().sum(0).float(),
            "n_free_rows": rows,
            "sum_free_flips": flips.double().sum().float(),
            "n_free_flip_pairs": both.double().sum().float(),
        }

    # -- checkpointing: add the decoder's architecture + n_ticks/n_codes/context_dim ----
    def save_pretrained(self, output_dir: Union[str, Path]):
        super().save_pretrained(output_dir)
        path = Path(output_dir) / HEAD_CONFIG_FILE
        meta = json.loads(path.read_text())
        meta["ar_n_ticks"] = self.n_ticks
        meta["ar_n_codes"] = self.n_codes
        meta["ar_context_dim"] = int(self.target_shape[0])
        meta["ar_decoder_kwargs"] = self.decoder.to_config()
        path.write_text(json.dumps(meta, indent=2))

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: Union[str, Path],
        processor,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
        **overrides,
    ) -> "TurnARActionClassifier":
        checkpoint_dir = Path(checkpoint_dir)
        meta = json.loads((checkpoint_dir / HEAD_CONFIG_FILE).read_text())
        model_cfg = ModelConfig(**{**migrate_model_config(meta["model"]), **overrides})
        loss_cfg = LossConfig(**meta["loss"])
        model = cls.build(
            model_cfg, loss_cfg, lora=None, n_ticks=meta["ar_n_ticks"], processor=processor,
            n_codes=meta["ar_n_codes"], context_dim=meta["ar_context_dim"],
            decoder_kwargs=meta.get("ar_decoder_kwargs"), dtype=dtype,
        )
        model.train_content_len = meta.get("train_content_len")
        adapter_dir = checkpoint_dir / ADAPTER_SUBDIR
        if adapter_dir.exists():
            from peft import PeftModel

            model.backbone = PeftModel.from_pretrained(model.backbone, str(adapter_dir))
        model.load_trainable(checkpoint_dir, adapter=False)
        if device:
            model.to(device)
        return model.eval()


# ======================================================================================
# Trainer: same band-statistic bookkeeping pattern as BinSFTTrainer, D=3 (dx, dy, dtheta)
# ======================================================================================
AR_METRIC_KEYS = ("sum_correct", "sum_stop_pred", "sum_stop_gt", "sum_creep_pred",
                  "sum_creep_gt", "sum_topk",
                  # free-running (eval only; absent from training-step outputs)
                  "sum_free_correct", "sum_free_sq_err", "sum_free_abs_err",
                  "n_free_rows", "sum_free_flips", "n_free_flip_pairs")


class ARActionSFTTrainer(TurnVectorSFTTrainer):
    """`TurnVectorSFTTrainer` plus the band statistics, all-reduced and logged. A near
    copy of `bin_head.BinSFTTrainer`'s bookkeeping (deliberately not imported/subclassed:
    that class's `_drain_metrics` docstring and division baked in the 30-independent-slot
    framing, and duplicating the ~30 lines here keeps this file self-contained without
    editing bin_head.py)."""

    def _accumulate(self, outputs: Dict[str, torch.Tensor]):
        super()._accumulate(outputs)
        for key in AR_METRIC_KEYS:
            if key not in outputs:
                continue
            v = outputs[key].detach().float()
            self._sums[key] = v.clone() if key not in self._sums else self._sums[key] + v

    def _drain_metrics(self, prefix: str = "") -> Dict[str, float]:
        sums = dict(self._sums)
        out = super()._drain_metrics(prefix)
        if not out or "n_rows" not in sums:
            return out
        rows = float(sums["n_rows"].clamp(min=1))
        names = DIM_NAMES
        for key, label in (("sum_correct", "acc"), ("sum_topk", "top5"),
                           ("sum_stop_pred", "stop_pred"), ("sum_stop_gt", "stop_gt"),
                           ("sum_creep_pred", "creep_pred"), ("sum_creep_gt", "creep_gt")):
            if key not in sums:
                continue
            for name, v in zip(names, sums[key].tolist()):
                out[f"{prefix}{label}_{name}"] = v / rows
        if "sum_stop_pred" in sums and float(sums["sum_stop_gt"][0]) > 0:
            out[f"{prefix}stop_ratio_dx"] = float(
                sums["sum_stop_pred"][0] / sums["sum_stop_gt"][0]
            )

        # -- free-running (eval only) ---------------------------------------------------
        # Deliberately NOT folded into the loop above: these divide by their own row
        # counter, and the flip rate divides by active PAIRS rather than rows.
        if "n_free_rows" in sums:
            frows = float(sums["n_free_rows"].clamp(min=1))
            for name, v in zip(names, sums["sum_free_correct"].tolist()):
                out[f"{prefix}free_acc_{name}"] = v / frows
            rmse = (sums["sum_free_sq_err"] / frows).sqrt()
            mae = sums["sum_free_abs_err"] / frows
            for name, r, m in zip(names, rmse.tolist(), mae.tolist()):
                out[f"{prefix}free_rmse_{name}"] = r
                out[f"{prefix}free_mae_{name}"] = m
            out[f"{prefix}free_rmse_mean"] = float(rmse.mean())
            pairs = float(sums["n_free_flip_pairs"])
            if pairs > 0:
                out[f"{prefix}free_rotation_flip"] = float(sums["sum_free_flips"]) / pairs
        return out
