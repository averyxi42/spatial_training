"""
A discrete-bin, cross-entropy head for action chunks -- the *control* for the flow work.

The question this exists to answer is narrow: the policy creeps because an MSE head
predicts the conditional mean of a stop-or-drive mixture and lands in the gap between the
modes (`dump/data_diagnostics/FINDINGS.md`). Does *any* distributional head fix that, or
is the conditional normalizing flow specifically needed? So this is the crudest reasonable
distributional head -- independent categoricals over discretized per-tick differentials,
trained with cross-entropy -- run against the same data, backbone, optimizer and metrics
as the regression baseline, changing only the output parameterization.

It is not a candidate for the product, and its weaknesses are load-bearing for its role as
a control: discretization aliases, and 30 independent categoricals model the joint
distribution over a chunk badly (nothing stops the head from putting mass on "stopped at
tick 3" and "driving at tick 4" in a combination that never occurs). Making it
autoregressive would fix the second and stop it being a control.

What changes relative to `vector_sft.TurnVectorRegressor`
---------------------------------------------------------
Only three things, all local:

  * `target_shape` gains a bin axis -- `(T, 3)` becomes `(T, 3, n_bins)` -- so the shared
    `out_dim = prod(target_shape)` arithmetic sizes the head's final projection to logits
    with no other change.
  * the `normalizer` slot holds a `BinCodec` instead of a `TargetNormalizer`. That slot's
    contract is "convert between model space and target units", which is exactly what the
    codec does, so `save_pretrained`, `load_head_state` and -- importantly --
    `VectorRolloutPolicy.step` all work untouched: the policy still calls `head(...)` then
    `normalizer.denormalize(...)` and still gets a chunk in metres and radians.
  * `forward` computes cross-entropy over the `T x 3` categoricals instead of Huber.

`vector_sft.py`, `turn_vectors.py` and `vector_rollout.py` are not modified.

Metrics
-------
The training loop reports the statistic the whole investigation turns on, live: the
fraction of predicted ticks that land in the exact-stop bin, against the data's. If the
answer to the framing question is "yes, any distributional head fixes it", that number
goes to ~0.6-0.7 within a few hundred steps and stays; the regression head sat at
0.002-0.012 at every checkpoint from 250 to 6500.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from longnav.utils.bin_codec import DIM_NAMES, BinCodec, decompose_chunk
from longnav.utils.vector_sft import (
    LossConfig,
    LoraSpec,
    ModelConfig,
    TurnVectorRegressor,
    TurnVectorSFTTrainer,
)
from longnav.utils.turn_vectors import extract_turn_vectors

# The band edges the existing analysis measures against (`sweep_analysis.py`), so the
# numbers printed during training are on the same footing as the sweep's.
EXACT = 1e-4
CREEP = (0.01, 0.01, 0.02)  # dx, dy, dtheta

# Extra accumulators this head reports on top of the base trainer's.
BIN_METRIC_KEYS = ("sum_correct", "sum_stop_pred", "sum_stop_gt",
                   "sum_creep_pred", "sum_creep_gt", "sum_topk")


class TurnBinClassifier(TurnVectorRegressor):
    """`TurnVectorRegressor` with a categorical output per (tick, dimension)."""

    # -- construction ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        model_cfg: ModelConfig,
        loss_cfg: LossConfig,
        lora: Optional[LoraSpec],
        target_shape: Sequence[int],
        processor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "TurnBinClassifier":
        """`target_shape` is `(T, n_dims, n_bins)`; the trailing axis is the bin axis."""
        target_shape = tuple(int(d) for d in target_shape)
        if len(target_shape) != 3:
            raise ValueError(
                f"target_shape must be (T, n_dims, n_bins), got {target_shape}. The bin "
                "axis is what sizes the head's output projection."
            )
        # The parent builds a TargetNormalizer over `target_shape[-1]`; with
        # normalize_targets off that object is inert, and it is replaced below. Forcing
        # the flag here means a stray --no-target-norm on the command line cannot make
        # the parent raise on an un-fitted normalizer we never use.
        loss_cfg = LossConfig(**{**loss_cfg.__dict__, "normalize_targets": False})
        model = super().build(model_cfg, loss_cfg, lora, target_shape, processor, dtype)
        model.normalizer = BinCodec(n_dims=target_shape[1], n_bins=target_shape[2])
        return model

    @property
    def codec(self) -> BinCodec:
        return self.normalizer

    @property
    def n_bins(self) -> int:
        return self.target_shape[-1]

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
        vectors, spans = extract_turn_vectors(
            outputs,
            input_ids,
            self.head,
            prefix_ids=self.prefix_ids,
            postfix_ids=self.postfix_ids,
            shift_left=self.model_cfg.shift_left,
            strict=True,
        )
        if vectors.shape[0] != targets.shape[0]:
            tail = input_ids[0, -64:].tolist()
            raise RuntimeError(
                f"found {vectors.shape[0]} assistant turn span(s) but got "
                f"{targets.shape[0]} target(s). prefix={self.model_cfg.prefix!r} "
                f"postfix={self.model_cfg.postfix!r} "
                f"shift_left={self.model_cfg.shift_left}. Last 64 input_ids: {tail}"
            )
        if self.train_content_len is None and spans:
            self.train_content_len = int(len(spans[0]))

        T, D, K = self.target_shape
        logits = vectors.view(-1, T, D, K).float()
        targets = targets.to(logits.device)
        gt_diffs = decompose_chunk(targets.double())
        labels = self.codec.encode_diffs(gt_diffs)                    # (N, T, D) long

        # Mean cross-entropy over the T x D categoricals, then summed over turns and
        # divided by the batch-wide turn count -- the same normalization the regression
        # head uses, so `train/turn_loss` curves are read the same way (different units).
        ce = F.cross_entropy(
            logits.reshape(-1, K), labels.reshape(-1), reduction="none"
        ).view(-1, T * D)
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
            metrics["n_dense_tokens"] = torch.tensor(
                input_ids.shape[1], device=logits.device
            )
            metrics["n_steps"] = torch.tensor(1, device=logits.device)
        return {"loss": loss, **metrics}

    @torch.no_grad()
    def _band_metrics(self, logits, labels, gt_diffs) -> Dict[str, torch.Tensor]:
        """Accuracy plus the exact-stop / creep band masses, per dimension.

        Everything is a per-dimension SUM over ticks so the trainer can all-reduce it and
        divide by the tick count once, rather than averaging averages across ranks.
        """
        D = self.target_shape[1]
        pred = logits.argmax(-1)                                   # (N, T, D)
        pred_diffs = self.codec.decode_labels(pred)
        dev = logits.device
        creep = torch.tensor(CREEP[:D], dtype=torch.float64, device=dev)

        def bands(v):
            a = v.abs()
            stop = (a < EXACT).reshape(-1, D).double().sum(0)
            crp = ((a >= EXACT) & (a < creep)).reshape(-1, D).double().sum(0)
            return stop, crp

        stop_p, creep_p = bands(pred_diffs)
        stop_g, creep_g = bands(gt_diffs)
        err = (pred_diffs - gt_diffs).reshape(-1, D)
        # Top-5 as a sanity check that a wrong argmax is still a near miss rather than a
        # confidently wrong bin -- with 65 bins, top-1 alone is a harsh and noisy read.
        topk = (logits.topk(min(5, logits.shape[-1]), dim=-1).indices
                == labels.unsqueeze(-1)).any(-1)
        return {
            "sum_sq_err": err.pow(2).sum(0).float(),
            "sum_abs_err": err.abs().sum(0).float(),
            "n_rows": torch.tensor(err.shape[0], device=dev),
            "sum_correct": (pred == labels).reshape(-1, D).double().sum(0).float(),
            "sum_topk": topk.reshape(-1, D).double().sum(0).float(),
            "sum_stop_pred": stop_p.float(),
            "sum_stop_gt": stop_g.float(),
            "sum_creep_pred": creep_p.float(),
            "sum_creep_gt": creep_g.float(),
        }


class BinSFTTrainer(TurnVectorSFTTrainer):
    """`TurnVectorSFTTrainer` plus the band statistics, all-reduced and logged."""

    def _accumulate(self, outputs: Dict[str, torch.Tensor]):
        super()._accumulate(outputs)
        for key in BIN_METRIC_KEYS:
            if key not in outputs:
                continue
            v = outputs[key].detach().float()
            self._sums[key] = v.clone() if key not in self._sums else self._sums[key] + v

    def _drain_metrics(self, prefix: str = "") -> Dict[str, float]:
        sums = dict(self._sums)          # the parent clears `self._sums`
        out = super()._drain_metrics(prefix)
        if not out or "n_rows" not in sums:
            return out
        rows = float(sums["n_rows"].clamp(min=1))
        names = self._dim_names(int(sums["sum_stop_pred"].numel()))
        for key, label in (("sum_correct", "acc"), ("sum_topk", "top5"),
                           ("sum_stop_pred", "stop_pred"), ("sum_stop_gt", "stop_gt"),
                           ("sum_creep_pred", "creep_pred"), ("sum_creep_gt", "creep_gt")):
            if key not in sums:
                continue
            for name, v in zip(names, sums[key].tolist()):
                out[f"{prefix}{label}_{name}"] = v / rows
        # The single number the control is about: predicted stop mass on forward motion,
        # as a fraction of the data's. 1.0 means the head commits to stops as often as
        # the robot does; the regression head sits at 0.003-0.016 of it.
        if "sum_stop_pred" in sums and float(sums["sum_stop_gt"][0]) > 0:
            out[f"{prefix}stop_ratio_{names[0]}"] = float(
                sums["sum_stop_pred"][0] / sums["sum_stop_gt"][0]
            )
        return out
