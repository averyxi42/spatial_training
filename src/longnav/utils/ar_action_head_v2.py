"""AR action head, version 2: the same objective, with the v1 decoder's accumulated warts
removed.

WHY A FORK RATHER THAN MORE EDITS. `ar_action_head.py` reached the point where every
remaining defect could only be fixed by changing what a checkpoint's weights MEAN, and two
runs (`run_v2_lloyd_ddp4`, `run_v3_ctx8_pose`) depend on those meanings. Each fix therefore
had to ship as an opt-in flag defaulting to the broken behaviour -- `pose_scales` defaulting
to LEGACY, `direct_context` defaulting to False, `start_embed` retained purely so old state
dicts load. That is the correct call for a live run and the wrong shape for a module anyone
has to read. v1 stays frozen and correct for its checkpoints; v2 makes the choices v1 could
only offer.

WHAT IS FORKED, AND WHAT IS DELIBERATELY NOT. Only the decoder is a mess, so only the
decoder is rewritten. `ChunkVQCodec`, `ARActionCodec`, `TurnARActionClassifier`'s metric /
save / load machinery and `ARActionSFTTrainer` are imported from v1 unchanged -- they are
clean, they are well covered, and forking them would fork the free-running metrics and the
dy fix along with them. `TurnARActionClassifierV2` is a two-line subclass that swaps
`DECODER_CLS`.

--------------------------------------------------------------------------------------
What v2 changes, and why (each traceable to a measurement)
--------------------------------------------------------------------------------------
1. NO `context_proj`. The readout MLP emits `n_context_tokens * d_model` and the decoder
   only reshapes. In v1 this projection sat immediately after the readout MLP's own final
   Linear with no activation between them, so the pair collapsed to one linear map and
   ~1.05M parameters did nothing. v1 fixed this behind `--direct-context`; here it is the
   only behaviour.

2. NO `start_embed`. It duplicated `context_proj.bias`, and with the projection gone it
   duplicates the readout MLP's final bias, which is already per-slot after the reshape.
   The slots remain distinguishable because each reads its own block of the MLP's output.

3. STATE CHANNEL REBUILT. v1 fed 7 features: 4 pose + 3 that restated the previous
   differential already carried categorically by `token_embed(label_{t-1})` -- measured
   correlation 0.83/0.77 with the pose features, and identical to them at tick 1 by
   construction. v2 keeps the accumulated pose only:

     [x/S_XY, y/S_XY, theta/S_TH]                       always
     [dx_in/S_DXY, dy_in/S_DXY, dth_in/S_DTH, valid]    iff use_incoming_motion

   * `theta` raw, not `(cos, sin)`. Accumulated heading inside a chunk never approaches
     +-pi, so there is no wraparound to protect against, and v1's `cos` was measured
     near-constant (std 0.058 against sin's 0.259) -- one wasted feature of two.
   * per-feature scales, calibrated to ~unit variance on v2_25hz. v1 used ONE constant for
     both accumulated pose and per-tick differentials, whose ranges differ ~9x, leaving a
     ~21x spread in feature scale (x std 1.82 against dy_prev 0.087, |max| 7.2).
   * a LayerNorm on the state before projection, so the scaling constants stop being
     load-bearing at all. v1 had none, unlike the readout MLP's own `pre_norm`.

4. THE INCOMING-MOTION SLOT, off by default. v1's tick 0 was told the robot is at rest --
   `BOS` token, zero pose, zero velocity -- when under a 50%-overlapping receding horizon
   it is mid-motion, carrying the velocity of the last tick it executed. Measured cost:
   teacher-forced accuracy 0.4254 at tick 0 against ~0.73 at every other tick, a ~30pp gap
   whose only structural cause is the missing previous differential (lag-1 R^2 of the
   differentials is 0.9566). `use_incoming_motion` reserves the feature slots plus a
   VALIDITY BIT -- without that bit an all-zero vector conflates "genuinely stationary"
   with "unknown", which imply opposite continuations. It defaults OFF because the value
   still has to be plumbed through the dataset and collator
   (`decompose_chunk(action_chunks[i-1])[stride-1]`); the architecture is ready for it so
   that adding the data later is not another checkpoint break.

Everything else -- token/tick embeddings, the causal/markov masks, the cacheless sequential
decode and its placeholder-safety invariant -- is carried over unchanged, including the
`is_causal=False` requirement for any non-triangular mask.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from longnav.utils.ar_action_head import (  # noqa: F401  (re-exported for convenience)
    ARActionCodec,
    ARActionSFTTrainer,
    ChunkVQCodec,
    CREEP,
    DEFAULT_DT_NATIVE,
    DIM_NAMES,
    EXACT,
    FLIP_DEADBAND_RADPS,
    TurnARActionClassifier,
)
from longnav.utils.bin_codec import compose_chunk

HEAD_VERSION = 2


class ActionDecoderV2(nn.Module):
    """Causal transformer over per-tick VQ codes. Same contract as v1's
    `CausalActionDecoder` -- `forward(context, labels, centroids)` and
    `decode(context, mode, ..., centroids)` -- so `ARActionCodec` drives it unchanged.
    """

    #: (pose_xy, pose_theta, diff_xy, diff_theta), calibrated to ~unit variance on
    #: v2_25hz: measured stds x 0.182 m, y 0.054 m, theta 0.05 rad, dx 0.032 m, dth
    #: 0.051 rad. `pose_xy` must cover x and y jointly or the scaling is anisotropic and
    #: distorts the geometry the decoder reasons over; likewise `diff_xy`.
    SCALES = (0.15, 0.05, 0.03, 0.05)

    def __init__(self, context_dim: int, n_codes: int, n_ticks: int, d_model: int = 128,
                 n_layers: int = 4, n_heads: int = 4, dim_ff: int = 512,
                 dropout: float = 0.1, n_context_tokens: int = 8,
                 attn_mode: str = "causal", use_incoming_motion: bool = False,
                 scales: Optional[Sequence[float]] = None):
        super().__init__()
        if attn_mode not in ("causal", "markov"):
            raise ValueError(f"attn_mode must be causal|markov, got {attn_mode!r}")
        want = int(n_context_tokens) * d_model
        if context_dim != want:
            raise ValueError(
                f"v2 has no context projection: the readout MLP must emit exactly "
                f"n_context_tokens * d_model ({n_context_tokens} * {d_model} = {want}), "
                f"got context_dim={context_dim}"
            )
        self.n_codes, self.n_ticks, self.d_model = n_codes, n_ticks, d_model
        self.n_context_tokens = int(n_context_tokens)
        self.context_dim = int(context_dim)
        self.attn_mode = attn_mode
        self.use_incoming_motion = bool(use_incoming_motion)
        self.scales = tuple(float(x) for x in (scales or self.SCALES))
        if len(self.scales) != 4 or any(x <= 0 for x in self.scales):
            raise ValueError(f"scales must be 4 positive floats, got {scales!r}")
        self.BOS = n_codes
        self.state_dim = 7 if self.use_incoming_motion else 3

        self.token_embed = nn.Embedding(n_codes + 1, d_model)
        self.tick_embed = nn.Embedding(n_ticks, d_model)
        self.state_norm = nn.LayerNorm(self.state_dim)
        self.state_proj = nn.Linear(self.state_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_ff, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, n_codes)
        self._init_kwargs = dict(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads, dim_ff=dim_ff,
            dropout=dropout, n_context_tokens=self.n_context_tokens,
            attn_mode=self.attn_mode, use_incoming_motion=self.use_incoming_motion,
            scales=list(self.scales),
        )

    def to_config(self) -> Dict:
        return dict(self._init_kwargs)

    def _tick_embeds(self, device) -> torch.Tensor:
        return self.tick_embed(torch.arange(self.n_ticks, device=device))

    def _attn_mask(self, device, dtype) -> Tuple[torch.Tensor, bool]:
        """`causal`: lower-triangular. `markov`: each tick sees the context slots and
        itself only, which is a sufficient conditioning set because the state channel
        supplies the accumulated pose.

        `is_causal` is returned False for `markov`: it is a HINT that permits the fast path
        to apply plain causal masking, which would silently discard a custom mask.
        """
        C, T = self.n_context_tokens, self.n_ticks
        L = C + T
        if self.attn_mode == "causal":
            return nn.Transformer.generate_square_subsequent_mask(
                L, device=device, dtype=dtype), True
        allow = torch.zeros(L, L, dtype=torch.bool, device=device)
        allow[:, :C] = True
        allow[:C, :C] = True
        idx = torch.arange(C, L, device=device)
        allow[idx, idx] = True
        return torch.zeros(L, L, dtype=dtype, device=device).masked_fill(
            ~allow, float("-inf")), False

    def _state(self, labels: torch.Tensor, centroids: torch.Tensor,
               incoming: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(N, T, state_dim). Position t describes the state BEFORE tick t.

        Adds no information -- it is a deterministic function of the same prefix the model
        conditions on -- it removes a computation attention is poor at: composing SE(2) is
        a nonlinear recurrence (products of rotation matrices), not a weighted sum.

        Placeholder safety, the invariant `decode` depends on: position t's features are
        composed from `labels[:, :t]` only, so they are correct at the step they are read;
        the garbage composed at later positions is discarded by the mask.

        `incoming`: (N, 4) `[dx, dy, dtheta, valid]` describing motion carried INTO tick 0
        from the previous chunk. Only consulted when `use_incoming_motion`; absent means
        "unknown", encoded as zeros with valid=0 rather than silently as "at rest".
        """
        N, T = labels.shape
        diffs = centroids.to(labels.device).index_select(
            0, labels.reshape(-1)).reshape(N, T, 3).float()
        poses = compose_chunk(diffs.double()).float()
        z = poses.new_zeros(N, 1, 3)
        p_prev = torch.cat([z, poses[:, :-1, :]], dim=1)
        s_xy, s_th = self.scales[0], self.scales[1]
        feats = [p_prev[..., 0] / s_xy, p_prev[..., 1] / s_xy, p_prev[..., 2] / s_th]

        if self.use_incoming_motion:
            d_xy, d_th = self.scales[2], self.scales[3]
            if incoming is None:
                incoming = poses.new_zeros(N, 4)
            inc = incoming.to(poses.dtype)
            pad = poses.new_zeros(N, T - 1)
            # Only tick 0 carries it; later ticks get their predecessor via token_embed.
            feats += [
                torch.cat([(inc[:, 0] / d_xy).unsqueeze(1), pad], 1),
                torch.cat([(inc[:, 1] / d_xy).unsqueeze(1), pad], 1),
                torch.cat([(inc[:, 2] / d_th).unsqueeze(1), pad], 1),
                torch.cat([inc[:, 3].unsqueeze(1), pad], 1),
            ]
        return torch.stack(feats, dim=-1)

    def forward(self, context: torch.Tensor, labels: torch.Tensor,
                centroids: Optional[torch.Tensor] = None,
                incoming: Optional[torch.Tensor] = None) -> torch.Tensor:
        if centroids is None:
            raise ValueError("v2's state channel is not optional; pass `centroids`")
        N, T = labels.shape
        device, C = context.device, self.n_context_tokens
        ctx_tok = context.view(N, C, self.d_model)          # reshape only -- no projection

        bos = torch.full((N, 1), self.BOS, dtype=torch.long, device=device)
        prev = torch.cat([bos, labels[:, :-1]], dim=1)
        state = self._state(labels, centroids, incoming)
        tick_x = (self.token_embed(prev)
                  + self._tick_embeds(device).unsqueeze(0)
                  + self.state_proj(self.state_norm(state.to(context.dtype))))
        x = torch.cat([ctx_tok, tick_x], dim=1)

        mask, is_causal = self._attn_mask(device, x.dtype)
        h = self.blocks(x, mask=mask, is_causal=is_causal)
        return self.out_proj(self.ln_f(h[:, C:, :]))

    @torch.no_grad()
    def decode(self, context: torch.Tensor, mode: str = "sample",
               generator: Optional[torch.Generator] = None, temperature: float = 1.0,
               centroids: Optional[torch.Tensor] = None,
               incoming: Optional[torch.Tensor] = None):
        """Sequential decode, T passes, correct without a KV cache -- see `_state`."""
        if mode not in ("argmax", "sample", "mean"):
            raise ValueError(f"decode mode must be argmax/sample/mean, got {mode!r}")
        if mode == "mean":
            raise ValueError(
                "mode='mean' is not sequentially decodable (no single token to condition "
                "later ticks on); ARActionCodec.denormalize(decode='mean') computes it "
                "from one teacher-forced-style pass, for reporting only"
            )
        N, T = context.shape[0], self.n_ticks
        labels = torch.zeros(N, T, dtype=torch.long, device=context.device)
        logits_all = torch.zeros(N, T, self.n_codes, device=context.device,
                                 dtype=torch.float32)
        for t in range(T):
            logits = self.forward(context, labels, centroids=centroids, incoming=incoming)
            lt = logits[:, t, :]
            logits_all[:, t, :] = lt
            if mode == "argmax":
                tok = lt.argmax(-1)
            else:
                probs = (lt.float() / max(temperature, 1e-6)).softmax(-1)
                tok = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            labels[:, t] = tok
        return labels, logits_all


class TurnARActionClassifierV2(TurnARActionClassifier):
    """v1's classifier with v2's decoder. Everything else -- band metrics, free-running
    metrics, save/load -- is inherited deliberately, so improvements to those keep
    benefiting both heads."""

    DECODER_CLS = ActionDecoderV2

    def save_pretrained(self, output_dir):
        super().save_pretrained(output_dir)
        import json
        from pathlib import Path

        from longnav.utils.vector_sft import HEAD_CONFIG_FILE

        path = Path(output_dir) / HEAD_CONFIG_FILE
        meta = json.loads(path.read_text())
        meta["ar_head_version"] = HEAD_VERSION   # so a loader can dispatch on it
        path.write_text(json.dumps(meta, indent=2))
