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
        """(..., 3) differentials -> (...,) long codebook indices, nearest centroid."""
        self._require_fitted()
        v = diffs.double()
        flat = v.reshape(-1, self.n_dims)
        joint_zero = (flat.abs() < self.zero_tol).all(dim=-1)
        # (n, n_codes) squared distance to every centroid; small enough (n_codes <= a few
        # thousand) that a dense cdist is simpler and fast enough than a KD-tree.
        d2 = torch.cdist(flat, self.centroids) ** 2
        idx = d2.argmin(dim=-1)
        idx = torch.where(joint_zero, torch.zeros_like(idx), idx)
        return idx.reshape(v.shape[:-1])

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
        obj = (json.loads(Path(path_or_obj).read_text())
               if isinstance(path_or_obj, (str, Path)) else path_or_obj)
        cen = obj.get("centroids")
        n_codes = obj.get("n_codes", obj.get("n_clusters"))
        codec = cls(n_codes=n_codes, n_dims=len(cen[0]), zero_tol=obj["zero_tol"])
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

    def __init__(self, context_dim: int, n_codes: int, n_ticks: int, d_model: int = 128,
                 n_layers: int = 4, n_heads: int = 4, dim_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.n_codes, self.n_ticks, self.d_model = n_codes, n_ticks, d_model
        self.BOS = n_codes  # one extra id: "no previous tick" (used only at position 1)

        self.context_proj = nn.Linear(context_dim, d_model)
        self.start_embed = nn.Parameter(torch.zeros(d_model))
        self.token_embed = nn.Embedding(n_codes + 1, d_model)
        self.tick_embed = nn.Embedding(n_ticks, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_ff, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, n_codes)
        # kept for checkpoint round-tripping (TurnARActionClassifier.save_pretrained)
        self._init_kwargs = dict(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                                 dim_ff=dim_ff, dropout=dropout)

    def to_config(self) -> Dict:
        return dict(self._init_kwargs)

    def _tick_embeds(self, device) -> torch.Tensor:
        return self.tick_embed(torch.arange(self.n_ticks, device=device))  # (T, d_model)

    def forward(self, context: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Teacher-forced. `context`: (N, context_dim) float. `labels`: (N, T) long, the
        TRUE code at every tick. Returns (N, T, n_codes) logits, position t predicting
        `labels[:, t]`."""
        N, T = labels.shape
        device = context.device
        ctx_tok = (self.context_proj(context) + self.start_embed).unsqueeze(1)  # (N,1,d)

        bos = torch.full((N, 1), self.BOS, dtype=torch.long, device=device)
        prev = torch.cat([bos, labels[:, :-1]], dim=1)               # (N, T)
        tok_e = self.token_embed(prev)                               # (N, T, d)
        slot_e = self._tick_embeds(device).unsqueeze(0)               # (1, T, d)
        x = torch.cat([ctx_tok, tok_e + slot_e], dim=1)               # (N, T+1, d)

        mask = nn.Transformer.generate_square_subsequent_mask(T + 1, device=device,
                                                               dtype=x.dtype)
        h = self.blocks(x, mask=mask, is_causal=True)
        return self.out_proj(self.ln_f(h[:, 1:, :]))                  # (N, T, n_codes)

    @torch.no_grad()
    def decode(self, context: torch.Tensor, mode: str = "sample",
              generator: Optional[torch.Generator] = None,
              temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequential decode, T forward passes. Correct WITHOUT a KV cache: at step `t`
        the input tensor is `labels` with positions `>= t` still zero-filled placeholders,
        but the causal mask means position `t`'s output depends only on positions `<= t`
        (i.e. only on already-decided ticks), so those placeholders can never leak into
        the logits actually used. Recomputing the full (tiny) sequence each step is
        `O(T^2)` in attention work, trivial at `T ~ 10` and simpler than incremental
        caching -- exactly the "genuinely tiny" brief this component is held to; a real
        deployment-scale version would want a cache, this prototype does not need one to
        be correct or fast enough to evaluate.
        """
        if mode not in ("argmax", "sample", "mean"):
            raise ValueError(f"decode mode must be argmax/sample/mean, got {mode!r}")
        N, T = context.shape[0], self.n_ticks
        device = context.device
        labels = torch.zeros(N, T, dtype=torch.long, device=device)
        logits_all = torch.zeros(N, T, self.n_codes, device=device, dtype=torch.float32)
        for t in range(T):
            logits = self.forward(context, labels)          # (N, T, n_codes)
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
        if self.decode == "mean":
            tokens, _ = self.decoder.decode(context, mode="argmax")
            logits = self.decoder.forward(context, tokens)
            probs = logits.float().softmax(-1)
            diffs = self.codec.decode_probs(probs)
        else:
            tokens, _ = self.decoder.decode(
                context, mode=self.decode, generator=self.generator,
                temperature=self.temperature,
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
        decoder = CausalActionDecoder(context_dim=context_dim, n_codes=n_codes,
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

        logits = self.decoder(context.float(), labels)           # (N, T, n_codes)
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

        used = [0, 2]  # dx, dtheta -- forward/rotation, matching every other head's table
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

        # Joint per-tick token correctness/top-5, reported once and duplicated across both
        # "dims" for interface symmetry with the bin head's table -- there is only one
        # categorical draw per tick, so both channels' correctness rise and fall together
        # by construction (documented, not a bug: see the module docstring).
        correct = (pred == labels).double().mean() * gt_diffs.shape[0] * gt_diffs.shape[1]
        topk = (logits.topk(min(5, logits.shape[-1]), dim=-1).indices
                == labels.unsqueeze(-1)).any(-1).double().sum()
        rows = torch.tensor(gt_diffs.shape[0] * gt_diffs.shape[1], device=dev)

        return {
            "sum_sq_err": err.pow(2).sum(0).float(),
            "sum_abs_err": err.abs().sum(0).float(),
            "n_rows": rows,
            "sum_correct": torch.stack([correct, correct]).float(),
            "sum_topk": torch.stack([topk, topk]).float(),
            "sum_stop_pred": stop_p.float(),
            "sum_stop_gt": stop_g.float(),
            "sum_creep_pred": creep_p.float(),
            "sum_creep_gt": creep_g.float(),
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
# Trainer: same band-statistic bookkeeping pattern as BinSFTTrainer, D=2 (dx, dtheta)
# ======================================================================================
AR_METRIC_KEYS = ("sum_correct", "sum_stop_pred", "sum_stop_gt", "sum_creep_pred",
                  "sum_creep_gt", "sum_topk")


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
        names = ("dx", "dtheta")
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
        return out
