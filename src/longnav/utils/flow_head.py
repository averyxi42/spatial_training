"""
A conditional-normalizing-flow head for action chunks -- the candidate replacement.

The regression head predicts the conditional *mean* of a stop-or-drive mixture and lands
in the gap between the modes; that is what creeping is (`dump/data_diagnostics/FINDINGS.md`).
This head predicts the conditional *density* instead, by maximum likelihood, so it can be
multimodal and commit. The flow itself is `longnav.cnf_head.flow.ConditionalFlow` -- read
its docstring first, it is the specification. This module is only the wiring that makes it
a head on the sparse Qwen3-VL trunk.

What changes relative to `vector_sft.TurnVectorRegressor`
---------------------------------------------------------
The same three local changes the bin-head control made, for the same reasons:

  * `target_shape` becomes `(context_dim,)`. The trunk no longer emits an action -- it
    emits the vector the density is *conditioned on*. `out_dim = prod(target_shape)`
    sizes the projection with no other change, and the conditioning trunk is the same
    width as the baseline's and the control's, so the three heads see the same
    information bottleneck.
  * the `normalizer` slot holds a `FlowActionDecoder`. That slot's contract is "convert
    between model space and target units", which is what it does: context vector in,
    action chunk in metres and radians out. `save_pretrained`, `load_head_state` and
    `VectorRolloutPolicy.step` therefore work untouched.
  * `forward` computes the flow's NLL over per-tick differentials instead of Huber over
    anchor-relative poses.

`vector_sft.py`, `turn_vectors.py` and `vector_rollout.py` are **not** modified.

Two things this head has that the other two do not
--------------------------------------------------
**A noise floor with a logged bound.** 33% of ticks are a bitwise-zero atom, and a
continuous density on an atom has unbounded likelihood. Training is on `x + sigma*eps`
with `sigma = 1e-5` (the measured width of the data's own float-residue cloud; it moves no
band statistic), annealed from 100x over the first steps. `flow/min_nll_per_dim` and
`flow/nll_margin` are logged every step: a run that crosses the floor is diverging, and
that is visible in seconds rather than at the end.

**A readout that is a choice.** The regression head's output is its prediction. A
density's is not: `best_of_k` commits to a mode, while averaging samples reproduces the
conditional mean and therefore reproduces the creeping exactly. Both are computed during
training and logged side by side (`stop_bok_*` vs `stop_mean_*`), because the difference
between them is the entire mechanism under test, and a head that only ever reported the
committed number would be assuming its own conclusion.

Autocast is disabled around every flow computation. The trainer runs bf16 autocast, and a
log-determinant accumulated over 12 layers in bf16 is not a log-determinant.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from longnav.cnf_head.flow import ConditionalFlow, flow_nll
from longnav.utils.bin_codec import compose_chunk, decompose_chunk
from longnav.utils.turn_vectors import extract_turn_vectors
from longnav.utils.vector_sft import (
    LossConfig,
    LoraSpec,
    ModelConfig,
    TurnVectorRegressor,
    TurnVectorSFTTrainer,
)

FLOW_CONFIG_FILE = "flow_config.json"

# The band edges every existing diagnostic uses (`dump/data_diagnostics/sweep_analysis.py`),
# so numbers printed during training sit on the same footing as the checkpoint sweep's.
EXACT = 1e-4
CREEP = (0.01, 0.01, 0.02)  # dx, dy, dtheta

FLOW_METRIC_KEYS = (
    "sum_stop_gt", "sum_creep_gt",
    "sum_stop_bok", "sum_creep_bok",
    "sum_stop_sample", "sum_creep_sample",
    "sum_stop_mean", "sum_creep_mean",
    "nll_floor_sum", "nll_sum_raw", "logdet_absmax", "sat_sum",
)

DECODE_RULES = ("best_of_k", "sample", "mode", "mean", "context")


def sigma_schedule(step: int, anneal_steps: int, start: float, end: float = 1.0) -> float:
    """Geometric anneal of the noise-floor multiplier, `start` -> `end`.

    Geometric because the quantity that matters -- the log-density bound the floor buys --
    is linear in log sigma, so a geometric schedule tightens the bound at a constant rate.
    """
    if anneal_steps <= 0 or step >= anneal_steps:
        return end
    return float(start * (end / start) ** (step / max(1, anneal_steps)))


class FlowActionDecoder(nn.Module):
    """Context vector -> action chunk. Occupies `TurnVectorRegressor`'s normalizer slot.

    `decode` picks the readout rule and is a plain attribute, so an evaluation script can
    flip it without rebuilding anything (`predict_flow.py` sets it to `"context"` and
    decodes offline, exactly as `predict_bins.py` sets the codec to `"logits"`).

      best_of_k  highest-log-density of `k` seeded draws -- the committed readout, and the
                 deterministic one. Seeded rather than merely stochastic because an
                 irreproducible closed-loop success rate is not a measurement.
      mode       the base mode pushed forward (eps = 0). Cheap; not the pushforward's mode.
      sample     a single draw -- what an on-policy RL rollout would execute.
      mean       the average of `k` draws. This is the conditional-mean readout, i.e. the
                 regression head's estimator computed from a distributional model. It is
                 here to be *measured*, not used: if the flow fixes creeping, this rule
                 should still creep.
      context    passthrough, for saving conditioning vectors and decoding offline.
    """

    def __init__(self, flow: ConditionalFlow, decode: str = "best_of_k",
                 k: int = 16, seed: int = 0, temperature: float = 1.0):
        super().__init__()
        self.flow = flow
        self.decode = decode
        self.k = int(k)
        self.seed = int(seed)
        self.temperature = float(temperature)

    # The slot's contract. Targets are never converted on the way in -- the flow's
    # objective is a density on the raw differentials, so there is nothing to normalize.
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.no_grad()
    def denormalize(self, ctx: torch.Tensor) -> torch.Tensor:
        """`ctx` (B, context_dim) -> anchor-relative chunk (B, T, 3) in metres/radians."""
        if self.decode == "context":
            return ctx
        ctx = ctx.float()
        with torch.autocast(device_type=ctx.device.type, enabled=False):
            diffs = self.decode_diffs(ctx, self.decode)
        return compose_chunk(diffs.double()).to(ctx.dtype)

    @torch.no_grad()
    def decode_diffs(self, ctx: torch.Tensor, rule: Optional[str] = None,
                     k: Optional[int] = None, seed: Optional[int] = None,
                     temperature: Optional[float] = None) -> torch.Tensor:
        """`ctx` -> per-tick differentials (B, T, 3), by the named rule."""
        rule = rule or self.decode
        k = self.k if k is None else int(k)
        seed = self.seed if seed is None else int(seed)
        temp = self.temperature if temperature is None else float(temperature)
        ctx = ctx.float()
        if rule == "best_of_k":
            return self.flow.best_of_k(ctx, k=k, seed=seed, temperature=temp)
        if rule == "mode":
            return self.flow.mode(ctx)
        g = torch.Generator(device=ctx.device).manual_seed(seed)
        if rule == "sample":
            x, _ = self.flow.sample(ctx, temperature=temp, generator=g)
            return x
        if rule == "mean":
            rep = ctx.repeat_interleave(k, dim=0)
            x, _ = self.flow.sample(rep, temperature=temp, generator=g)
            return x.view(ctx.shape[0], k, self.flow.chunk_len, self.flow.n_channels).mean(1)
        raise ValueError(f"unknown decode rule {rule!r}; expected one of {DECODE_RULES}")


class TurnFlowPolicy(TurnVectorRegressor):
    """`TurnVectorRegressor` whose trunk emits a conditioning vector and whose loss is NLL."""

    @classmethod
    def build(
        cls,
        model_cfg: ModelConfig,
        loss_cfg: LossConfig,
        lora: Optional[LoraSpec],
        target_shape: Sequence[int],
        processor,
        dtype: torch.dtype = torch.bfloat16,
        chunk_shape: Sequence[int] = (10, 3),
        flow_kwargs: Optional[Dict[str, Any]] = None,
        decode: str = "best_of_k",
        decode_k: int = 16,
    ) -> "TurnFlowPolicy":
        """`target_shape` is `(context_dim,)`; `chunk_shape` is the action chunk `(T, 3)`."""
        target_shape = tuple(int(d) for d in target_shape)
        if len(target_shape) != 1:
            raise ValueError(
                f"target_shape must be (context_dim,), got {target_shape}. The trunk emits "
                "the conditioning vector, not the action -- the action comes from the flow."
            )
        chunk_shape = tuple(int(d) for d in chunk_shape)
        # `normalize_targets` is meaningless here (a density is fitted in raw units) and
        # leaving it on would make the parent raise on an un-fitted normalizer.
        loss_cfg = LossConfig(**{**loss_cfg.__dict__, "normalize_targets": False})
        model = super().build(model_cfg, loss_cfg, lora, target_shape, processor, dtype)
        flow = ConditionalFlow(
            context_dim=target_shape[0], chunk_len=chunk_shape[0],
            n_channels=chunk_shape[1], **(flow_kwargs or {}),
        )
        model.normalizer = FlowActionDecoder(flow, decode=decode, k=decode_k)
        model.chunk_shape = chunk_shape
        model.flow_kwargs = dict(flow_kwargs or {})
        model.sigma_mult = 1.0
        return model

    # -- convenience -------------------------------------------------------------------
    @property
    def flow(self) -> ConditionalFlow:
        return self.normalizer.flow

    @property
    def decoder(self) -> FlowActionDecoder:
        return self.normalizer

    def set_sigma_mult(self, value: float):
        self.sigma_mult = float(value)

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

        ctx = vectors.view(-1, self.target_shape[0]).float()
        targets = targets.to(ctx.device)
        # The flow lives on per-tick body-frame differentials, not the anchor-relative
        # poses the dataset stores: that is the only space where "the robot did not move"
        # is the origin in every coordinate. Composed in float64 -- the differences are
        # O(1e-8) against poses of O(1), so float32 would manufacture its own residue.
        gt_diffs = decompose_chunk(targets.double()).float()

        # bf16 autocast is on for the backbone; a 12-layer log-determinant is not
        # meaningful in bf16, so the density is always computed in fp32.
        with torch.autocast(device_type=ctx.device.type, enabled=False):
            per_turn, _ = flow_nll(self.flow, gt_diffs, ctx, sigma_mult=self.sigma_mult)

        total = per_turn.sum()
        denom = num_items_in_batch if num_items_in_batch is not None else per_turn.numel()
        denom = torch.as_tensor(denom, dtype=total.dtype, device=total.device).clamp(min=1)
        loss = total / denom

        with torch.no_grad():
            metrics = self._flow_metrics(ctx.detach(), gt_diffs)
            metrics["loss_sum"] = total.detach()
            metrics["nll_sum_raw"] = total.detach()
            metrics["nll_floor_sum"] = torch.tensor(
                self.flow.min_nll_per_dim(self.sigma_mult) * per_turn.shape[0],
                device=ctx.device, dtype=torch.float32)
            metrics["n_turns"] = torch.tensor(ctx.shape[0], device=ctx.device)
            metrics["n_tokens"] = torch.tensor(
                outputs["last_hidden_state"].shape[1], device=ctx.device)
            metrics["n_dense_tokens"] = torch.tensor(input_ids.shape[1], device=ctx.device)
            metrics["n_steps"] = torch.tensor(1, device=ctx.device)
        return {"loss": loss, **metrics}

    @torch.no_grad()
    def _flow_metrics(self, ctx: torch.Tensor, gt_diffs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Band masses per readout rule, plus the stability numbers.

        Three readouts are measured every step because they answer different questions:
        `bok` is what the policy would execute, `sample` is what RL would execute, and
        `mean` is what the estimator the regression head uses would do with this very same
        model. If the flow's advantage is real, `bok` and `sample` sit near the data's
        stop mass while `mean` does not.
        """
        dev = ctx.device
        D = gt_diffs.shape[-1]
        creep = torch.tensor(CREEP[:D], device=dev)

        def bands(v):
            a = v.abs().reshape(-1, D)
            return ((a < EXACT).float().sum(0),
                    ((a >= EXACT) & (a < creep)).float().sum(0))

        with torch.autocast(device_type=dev.type, enabled=False):
            dec = self.decoder
            bok = dec.decode_diffs(ctx, "best_of_k")
            smp = dec.decode_diffs(ctx, "sample")
            mean = dec.decode_diffs(ctx, "mean")
            rep = self.flow.stability_report(gt_diffs[: min(256, len(gt_diffs))], ctx[:256])

        stop_g, creep_g = bands(gt_diffs)
        stop_b, creep_b = bands(bok)
        stop_s, creep_s = bands(smp)
        stop_m, creep_m = bands(mean)
        err = (bok - gt_diffs).reshape(-1, D)
        return {
            "sum_sq_err": err.pow(2).sum(0).float(),
            "sum_abs_err": err.abs().sum(0).float(),
            "n_rows": torch.tensor(err.shape[0], device=dev),
            "sum_stop_gt": stop_g, "sum_creep_gt": creep_g,
            "sum_stop_bok": stop_b, "sum_creep_bok": creep_b,
            "sum_stop_sample": stop_s, "sum_creep_sample": creep_s,
            "sum_stop_mean": stop_m, "sum_creep_mean": creep_m,
            "logdet_absmax": torch.tensor(rep["logdet_abs_max"], device=dev),
            "sat_sum": torch.tensor(rep["scale_saturation"], device=dev),
        }

    # -- checkpointing -----------------------------------------------------------------
    def save_pretrained(self, output_dir: Union[str, Path]):
        """Parent's checkpoint (adapter + head + the `normalizer` slot, i.e. the flow),
        plus the flow's own architecture config so `from_pretrained` can rebuild it."""
        super().save_pretrained(output_dir)
        (Path(output_dir) / FLOW_CONFIG_FILE).write_text(json.dumps({
            "flow": self.flow.config(),
            "chunk_shape": list(self.chunk_shape),
            "decode": self.decoder.decode,
            "decode_k": self.decoder.k,
        }, indent=2))

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: Union[str, Path],
        processor,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
        **overrides,
    ) -> "TurnFlowPolicy":
        from longnav.utils.vector_sft import (
            ADAPTER_SUBDIR, HEAD_CONFIG_FILE, migrate_model_config,
        )

        checkpoint_dir = Path(checkpoint_dir)
        meta = json.loads((checkpoint_dir / HEAD_CONFIG_FILE).read_text())
        fmeta = json.loads((checkpoint_dir / FLOW_CONFIG_FILE).read_text())
        model_cfg = ModelConfig(**{**migrate_model_config(meta["model"]), **overrides})
        flow_cfg = dict(fmeta["flow"])
        # These three are positional in build(); the rest is the flow's own architecture.
        for key in ("context_dim", "chunk_len", "n_channels"):
            flow_cfg.pop(key, None)
        model = cls.build(
            model_cfg, LossConfig(**meta["loss"]), lora=None,
            target_shape=meta["target_shape"], processor=processor, dtype=dtype,
            chunk_shape=fmeta["chunk_shape"], flow_kwargs=flow_cfg,
            decode=fmeta.get("decode", "best_of_k"), decode_k=fmeta.get("decode_k", 16),
        )
        model.train_content_len = meta.get("train_content_len")
        if (checkpoint_dir / ADAPTER_SUBDIR).exists():
            from peft import PeftModel

            model.backbone = PeftModel.from_pretrained(
                model.backbone, str(checkpoint_dir / ADAPTER_SUBDIR))
        model.load_trainable(checkpoint_dir, adapter=False)
        if device:
            model.to(device)
        return model.eval()


class FlowSFTTrainer(TurnVectorSFTTrainer):
    """`TurnVectorSFTTrainer` plus the noise-floor anneal and the flow's own metrics.

    The anneal is driven from `state.global_step` here rather than from a callback so it
    is correct on a resumed run without any extra bookkeeping: the schedule is a pure
    function of the step, so a restart lands on the same sigma the run would have had.
    """

    def __init__(self, *args, sigma_start: float = 100.0, sigma_anneal_steps: int = 1000,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_start = float(sigma_start)
        self.sigma_anneal_steps = int(sigma_anneal_steps)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sm = sigma_schedule(self.state.global_step, self.sigma_anneal_steps, self.sigma_start)
        self.accelerator.unwrap_model(model).set_sigma_mult(sm)
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def _accumulate(self, outputs: Dict[str, torch.Tensor]):
        super()._accumulate(outputs)
        for key in FLOW_METRIC_KEYS:
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
        turns = float(sums["n_turns"].clamp(min=1))
        steps = float(sums.get("n_steps", torch.tensor(1.0)).clamp(min=1))
        names = self._dim_names(int(sums["sum_stop_gt"].numel()))
        for key, label in (("sum_stop_gt", "stop_gt"), ("sum_creep_gt", "creep_gt"),
                           ("sum_stop_bok", "stop_bok"), ("sum_creep_bok", "creep_bok"),
                           ("sum_stop_sample", "stop_sample"),
                           ("sum_creep_sample", "creep_sample"),
                           ("sum_stop_mean", "stop_mean"),
                           ("sum_creep_mean", "creep_mean")):
            if key not in sums:
                continue
            for name, v in zip(names, sums[key].tolist()):
                out[f"{prefix}{label}_{name}"] = v / rows
        # The bound the noise floor buys, and the distance to it. A run whose margin goes
        # negative is exploiting the atom, not learning, and everything downstream of that
        # point is worthless -- so this is logged every step, not sampled.
        if "nll_floor_sum" in sums:
            floor = float(sums["nll_floor_sum"]) / turns
            nll = float(sums["nll_sum_raw"]) / turns
            out[f"{prefix}min_nll_per_dim"] = floor
            out[f"{prefix}nll_per_dim"] = nll
            out[f"{prefix}nll_margin"] = nll - floor
            # Distance to the softer, *reachable* floor (the entropy of a pure smeared
            # atom, 0.5 nats above the hard bound). Margin says "is this a bug"; headroom
            # says "how much of the achievable likelihood is left".
            out[f"{prefix}nll_headroom"] = nll - (floor + 0.5)
        if "sat_sum" in sums:
            out[f"{prefix}scale_saturation"] = float(sums["sat_sum"]) / steps
            out[f"{prefix}logdet_absmax"] = float(sums["logdet_absmax"]) / steps
        # The single number the project turns on: committed stop mass on forward motion as
        # a fraction of the data's. The regression head sat at 0.003-0.016 of it.
        if "sum_stop_bok" in sums and float(sums["sum_stop_gt"][0]) > 0:
            out[f"{prefix}stop_ratio_{names[0]}"] = float(
                sums["sum_stop_bok"][0] / sums["sum_stop_gt"][0])
        return out
