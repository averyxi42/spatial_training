"""Per-module observability: is this submodule actually learning, and on what scale?

Loss curves are a report on the *objective*. They say nothing about the parts, and three
expensive defects in this project were invisible in the loss for thousands of steps:

  1. `run_v6_regress_pose` trained ~3600 steps with its `FourierSE2Encoder` frozen. The
     output layer is zero-initialised and the `max_norm` soft cap multiplies by
     `tanh(|y|/m) * m / (|y| + 1e-12)`, which is exactly 0 when `|y| == 0`; the backward
     carries the same factor, so the gradient into the encoder was exactly zero forever.
     Every checkpoint's encoder weights are bit-identical.
  2. The same cap saturates at the far end: `run_v5_ar_pose`'s encoder emits norms of
     12-19, and `tanh` returns exactly 1.0 in fp32 above 9.011, so the radial gradient is
     exactly zero there too.
  3. `run_v5_ar_pose`'s injected embedding reached norm ~16.5 against an ordinary token
     embedding's ~1.45 -- 11x and diverging -- with a train grad-norm median of 10.7
     against a comparable run's 1.9.

Each of the three is one logged scalar away from being obvious on the first logged step:
a per-module grad norm (1), an activation norm against the module's own cap (2), and an
activation norm against the token-embedding norm (3). This module logs all of them, plus
the cumulative weight delta from init, which is the direct "this module has not moved"
test and the one that cannot be argued with.

    from longnav.utils.model_metrics import add_model_metrics_args, attach_model_metrics
    add_model_metrics_args(parser)          # in the entry point's argparse
    attach_model_metrics(trainer, args=args)   # right after the Trainer is built

Everything here is strictly observational. It reads `.grad` and module outputs, never
writes a parameter, never touches the RNG, never changes the loss -- attaching it cannot
alter a run's trajectory, only its logged keys.

Cost. Nothing is computed on an un-armed step: the forward hooks return on a boolean
check, and the gradient sweep only runs at `on_pre_optimizer_step` on armed steps. A step
is armed if it is one of the first `first_n` (default 3) or `step % every == 0`
(default 25). On an armed step the work is a handful of `norm()`/`sum()` reductions over
modules that are small by construction, all kept as 0-d device tensors and materialised in
a single `.cpu()` at log time -- one host sync per logged step, not one per scalar.

What is watched, by default, is auto-discovered (see `auto_watch`): the root's direct
children that are not the frozen backbone, plus the entries of any `ModuleDict`/
`ModuleList` inside them. On `TurnVectorRegressor` that is exactly `head`, `stop_head`,
`modality_embedder` and each `modality_embedder.encoders.<key>` -- the fresh modules, the
ones with no published baseline and therefore the ones that fail silently. Name modules
explicitly with `--watch-modules` (dotted names or globs) when you want something else.

Two caveats worth knowing before reading a number:

  * `grad_norm` is measured at `on_pre_optimizer_step`, which HF calls *after* gradient
    clipping, so on a step that clipped it is scaled by the global clip factor.
    `grad_frac` (this module's share of the total trainable grad norm) is invariant to
    that and is the better cross-step comparison. Exact zero is exact zero either way,
    which is the detection that matters.
  * `weight_delta` is measured from the weights present at `on_train_begin`, so on a
    resumed run it is movement since the resume, not since the run's own step 0.

Two mechanics worth knowing rather than rediscovering. Under gradient checkpointing the
forward hooks fire again during the backward recomputation; every activation figure here
is a mean, a min or a fraction, so the duplicate contributes the same value and nothing
is skewed -- only the internal counts double. And under DDP every rank collects: the
gradients have already been all-reduced by `on_pre_optimizer_step` so the gradient
numbers agree across ranks, the activation numbers are that rank's own microbatches, and
nothing here performs a collective, so it cannot deadlock a rank.
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

__all__ = [
    "ModelMetricsConfig",
    "ModelMetrics",
    "ModelMetricsCallback",
    "add_model_metrics_args",
    "config_from_args",
    "attach_model_metrics",
    "auto_watch",
    "resolve_watch",
]


# Bounded nonlinearities saturate; rectified ones die. Both are gradient death, and both
# are reported as a fraction of units so the number means the same thing at any width.
_BOUNDED = (nn.Tanh, nn.Sigmoid, nn.Softsign, nn.Hardtanh, nn.Hardsigmoid)
_RECTIFIED = (nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.ELU, nn.LeakyReLU, nn.Mish)
_CONTAINERS = (nn.ModuleDict, nn.ModuleList)


# ---------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------
@dataclass
class ModelMetricsConfig:
    """Everything the collector needs. Defaults are the zero-configuration setting.

    `watch` is the one knob most runs touch: an empty tuple means auto-discovery, and a
    non-empty one is a list of dotted module names or `fnmatch` globs resolved against
    `model.named_modules()` (`"modality_embedder.encoders.*"`, `"*.lora_A.default"`).
    `exclude` is applied after, in the same language.
    """

    enabled: bool = True
    # Cadence. `every` counts optimizer steps; `first_n` forces the opening steps so that
    # a module that is dead from initialisation is visible immediately rather than at 25.
    every: int = 25
    first_n: int = 3
    watch: Tuple[str, ...] = ()
    exclude: Tuple[str, ...] = ()
    prefix: str = "model"

    # Metric families, each independently switchable.
    gradients: bool = True
    grad_zero_frac: bool = True
    weights: bool = True
    weight_delta: bool = True
    activations: bool = True
    nonlinearities: bool = True
    include_total: bool = True
    embed_reference: bool = True

    # Auto-discovery and safety rails.
    auto_max_params: int = 50_000_000  # a child bigger than this is "the backbone"
    max_modules: int = 48
    max_snapshot_params: int = 20_000_000  # per module, for the weight-delta snapshot
    embed_sample: int = 4096  # token-embedding rows sampled for the reference norm

    # Thresholds.
    sat_tol: float = 0.01  # within 1% of the bound counts as saturated
    dead_tol: float = 1e-3
    eps: float = 1e-12

    def resolved_watch(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        picked = (
            resolve_watch(model, self.watch) if self.watch else auto_watch(model, self)
        )
        if self.exclude:
            picked = [
                (n, m) for n, m in picked
                if not any(fnmatch.fnmatchcase(n, pat) for pat in self.exclude)
            ]
        return picked[: self.max_modules]


# ---------------------------------------------------------------------------------------
# module selection
# ---------------------------------------------------------------------------------------
def resolve_watch(model: nn.Module, patterns: Sequence[str]) -> List[Tuple[str, nn.Module]]:
    """Dotted names and globs -> `(name, module)`, in `named_modules()` order.

    An exact dotted name that matches nothing is an error: it is almost always a typo or a
    renamed attribute, and silently watching nothing is the failure mode this whole module
    exists to prevent.
    """
    named = dict(model.named_modules())
    out: Dict[str, nn.Module] = {}
    for pat in patterns:
        pat = pat.strip()
        if not pat:
            continue
        if any(ch in pat for ch in "*?["):
            hits = [n for n in named if fnmatch.fnmatchcase(n, pat)]
        else:
            hits = [pat] if pat in named else []
            if not hits:
                raise KeyError(
                    f"--watch-modules: no module named {pat!r}. Nearest: "
                    + ", ".join(sorted(named, key=lambda n: _dist(n, pat))[:5])
                )
        for n in hits:
            if n:
                out[n] = named[n]
    return [(n, m) for n, m in named.items() if n in out]


def _dist(a: str, b: str) -> int:
    return abs(len(a) - len(b)) + sum(x != y for x, y in zip(a, b))


def auto_watch(model: nn.Module, cfg: ModelMetricsConfig) -> List[Tuple[str, nn.Module]]:
    """The default watch set: the fresh, small modules, never the backbone.

    A direct child holding more than `auto_max_params` parameters is the pretrained
    backbone -- it has a published baseline, it is not what silently fails, and walking it
    would produce hundreds of LoRA keys. It is skipped whole. Everything else is watched,
    and any `ModuleDict`/`ModuleList` inside it is expanded to its entries so that one dead
    encoder among several is not averaged into invisibility.
    """
    picked: List[Tuple[str, nn.Module]] = []
    for name, child in model.named_children():
        n_params = sum(p.numel() for p in child.parameters())
        if n_params == 0 or n_params > cfg.auto_max_params:
            continue
        picked.append((name, child))
        for sub_name, sub in child.named_modules():
            if not isinstance(sub, _CONTAINERS):
                continue
            for leaf_name, leaf in sub.named_children():
                if any(True for _ in leaf.parameters()):
                    full = ".".join(x for x in (name, sub_name, leaf_name) if x)
                    picked.append((full, leaf))
    seen: Dict[int, str] = {}
    unique = []
    for name, mod in picked:
        if id(mod) in seen:
            continue
        seen[id(mod)] = name
        unique.append((name, mod))
    return unique


# ---------------------------------------------------------------------------------------
# accumulation
# ---------------------------------------------------------------------------------------
class _Bucket:
    """Per-module accumulators kept as 0-d device tensors -- no host sync until drain.

    Counts are Python ints (`numel()` is metadata, not a device read), so only the sums,
    minima and maxima ever touch the GPU, and they are stacked into one transfer at drain.
    """

    def __init__(self) -> None:
        self.sums: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, int] = {}

    def add(self, key: str, value: torch.Tensor, n: int = 0) -> None:
        v = value.detach().to(torch.float32).reshape(())
        self.sums[key] = v if key not in self.sums else self.sums[key] + v
        if n:
            self.counts[key] = self.counts.get(key, 0) + n

    def minimum(self, key: str, value: torch.Tensor) -> None:
        v = value.detach().to(torch.float32).reshape(())
        self.sums[key] = v if key not in self.sums else torch.minimum(self.sums[key], v)

    def maximum(self, key: str, value: torch.Tensor) -> None:
        v = value.detach().to(torch.float32).reshape(())
        self.sums[key] = v if key not in self.sums else torch.maximum(self.sums[key], v)

    def clear(self) -> None:
        self.sums.clear()
        self.counts.clear()


def _first_tensor(out: Any) -> Optional[torch.Tensor]:
    """The first floating-point tensor in a module's output, whatever it is wrapped in."""
    if torch.is_tensor(out):
        return out if out.is_floating_point() else None
    if isinstance(out, (tuple, list)):
        for item in out:
            t = _first_tensor(item)
            if t is not None:
                return t
        return None
    if hasattr(out, "items"):
        for _, item in out.items():
            t = _first_tensor(item)
            if t is not None:
                return t
    return None


def _row_norms(x: torch.Tensor) -> torch.Tensor:
    """Per-row L2 over the last dimension -- the scale an injected vector is judged on."""
    x = x.detach().to(torch.float32)
    if x.dim() == 0:
        return x.abs().reshape(1)
    if x.dim() == 1:
        return x.norm().reshape(1)
    return x.reshape(-1, x.shape[-1]).norm(dim=-1)


# ---------------------------------------------------------------------------------------
# the collector
# ---------------------------------------------------------------------------------------
class ModelMetrics:
    """Watches a set of submodules and produces a flat `{"model/<mod>/<metric>": float}`.

    Usable without a `Trainer`:

        mm = ModelMetrics(model, ModelMetricsConfig(watch=("encoder",)))
        mm.start()                 # snapshot init weights, register hooks
        for step in range(n):
            mm.arm(mm.should_collect(step))
            ...forward / backward...
            mm.collect_gradients()
            print(mm.drain())      # {} on an un-armed step
        mm.stop()
    """

    def __init__(self, model: nn.Module, config: Optional[ModelMetricsConfig] = None):
        self.model = model
        self.cfg = config or ModelMetricsConfig()
        self.watched: List[Tuple[str, nn.Module]] = []
        self._buckets: Dict[str, _Bucket] = {}
        self._handles: List[Any] = []
        self._init: Dict[str, Dict[str, torch.Tensor]] = {}
        self._init_norm: Dict[str, float] = {}
        self._armed = False
        self._pending: Dict[str, float] = {}
        self._started = False
        self._embed: Optional[nn.Embedding] = None
        self._embed_rows: Optional[torch.Tensor] = None
        self._static: Dict[str, float] = {}

    # -- lifecycle ---------------------------------------------------------------------
    def start(self) -> "ModelMetrics":
        """Resolve the watch set, snapshot init weights, register hooks. Idempotent.

        Deliberately called at `on_train_begin` rather than at construction: HF restores a
        `--resume-from` checkpoint before that point, and a snapshot taken earlier would
        make `weight_delta` report the distance from the checkpoint's *predecessor*.
        """
        if self._started or not self.cfg.enabled:
            return self
        self.watched = self.cfg.resolved_watch(self.model)
        self._buckets = {name: _Bucket() for name, _ in self.watched}
        for name, mod in self.watched:
            params = [p for p in mod.parameters() if p.requires_grad]
            self._static[f"{name}/n_params"] = float(sum(p.numel() for p in mod.parameters()))
            self._static[f"{name}/n_trainable"] = float(sum(p.numel() for p in params))
            if self.cfg.weight_delta:
                self._snapshot(name, mod)
        if self.cfg.activations or self.cfg.nonlinearities:
            self._register_hooks()
        if self.cfg.embed_reference:
            self._find_embedding()
        self._started = True
        return self

    def stop(self) -> None:
        """Detach every hook and drop the init snapshot. Safe to call twice."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001 - a removed handle must never fail teardown
                pass
        self._handles = []
        self._init.clear()
        for b in self._buckets.values():
            b.clear()
        self._armed = False
        self._started = False

    def _snapshot(self, name: str, mod: nn.Module) -> None:
        n = sum(p.numel() for p in mod.parameters())
        if n == 0 or n > self.cfg.max_snapshot_params:
            return
        with torch.no_grad():
            snap = {k: p.detach().to(torch.float32).clone() for k, p in mod.named_parameters()}
        self._init[name] = snap
        total = sum(float(v.pow(2).sum()) for v in snap.values())
        self._init_norm[name] = math.sqrt(total)

    def _find_embedding(self) -> None:
        """The token embedding matrix, as the yardstick for "is this vector a sane size".

        A fixed random sample of rows, not the whole table: the reference is a typical row
        norm, 4096 rows estimate it to well under a percent, and the full matrix is
        150k x d, which is not something to reduce over on every logged step.
        """
        emb = None
        getter = getattr(self.model, "get_input_embeddings", None)
        if callable(getter):
            try:
                emb = getter()
            except Exception:  # noqa: BLE001 - not every wrapper implements it
                emb = None
        if not isinstance(emb, nn.Embedding):
            best = 0
            for _, m in self.model.named_modules():
                if isinstance(m, nn.Embedding) and m.num_embeddings > best:
                    best, emb = m.num_embeddings, m
        if not isinstance(emb, nn.Embedding):
            return
        self._embed = emb
        n = min(int(self.cfg.embed_sample), int(emb.num_embeddings))
        g = torch.Generator().manual_seed(0)
        self._embed_rows = torch.randperm(emb.num_embeddings, generator=g)[:n]

    def _register_hooks(self) -> None:
        for name, mod in self.watched:
            bucket = self._buckets[name]
            if self.cfg.activations:
                self._handles.append(
                    mod.register_forward_hook(self._make_act_hook(bucket))
                )
            if self.cfg.nonlinearities:
                for _, sub in mod.named_modules():
                    if isinstance(sub, _BOUNDED):
                        self._handles.append(
                            sub.register_forward_hook(self._make_nonlin_hook(bucket, True))
                        )
                    elif isinstance(sub, _RECTIFIED):
                        self._handles.append(
                            sub.register_forward_hook(self._make_nonlin_hook(bucket, False))
                        )

    # -- hooks -------------------------------------------------------------------------
    def _make_act_hook(self, bucket: _Bucket):
        cfg = self.cfg

        def hook(module, inputs, output):
            if not self._armed:
                return
            t = _first_tensor(output)
            if t is None or t.numel() == 0:
                return
            with torch.no_grad():
                norms = _row_norms(t)
                k = int(norms.numel())
                bucket.add("act_sum", norms.sum(), n=k)
                bucket.minimum("act_min", norms.min())
                bucket.maximum("act_max", norms.max())
                # A module carrying its own `max_norm` soft cap tells us what saturation
                # means for it, so the "output pinned at the cap" failure is detectable
                # without this module knowing anything about that module.
                cap = getattr(module, "max_norm", None)
                if isinstance(cap, (int, float)) and cap > 0:
                    hit = (norms >= (1.0 - cfg.sat_tol) * float(cap)).sum()
                    bucket.add("cap_sat_sum", hit, n=k)
                    bucket.add("cap", torch.as_tensor(float(cap), device=norms.device), n=1)

        return hook

    def _make_nonlin_hook(self, bucket: _Bucket, bounded: bool):
        cfg = self.cfg

        def hook(module, inputs, output):
            if not self._armed:
                return
            t = _first_tensor(output)
            if t is None or t.numel() == 0:
                return
            with torch.no_grad():
                y = t.detach().to(torch.float32)
                k = int(y.numel())
                if bounded:
                    if isinstance(module, (nn.Sigmoid, nn.Hardsigmoid)):
                        hit = ((y <= cfg.sat_tol) | (y >= 1.0 - cfg.sat_tol)).sum()
                    else:
                        hit = (y.abs() >= 1.0 - cfg.sat_tol).sum()
                    bucket.add("nl_sat_sum", hit, n=k)
                else:
                    bucket.add("nl_dead_sum", (y <= cfg.dead_tol).sum(), n=k)

        return hook

    # -- cadence -----------------------------------------------------------------------
    def should_collect(self, step: int) -> bool:
        if not self.cfg.enabled or not self._started:
            return False
        if self.cfg.first_n and step <= int(self.cfg.first_n):
            return True
        every = max(1, int(self.cfg.every))
        return step % every == 0

    def arm(self, on: bool = True) -> None:
        self._armed = bool(on) and self._started

    @property
    def armed(self) -> bool:
        return self._armed

    # -- gradients ---------------------------------------------------------------------
    @torch.no_grad()
    def collect_gradients(self) -> None:
        """Per-module gradient and weight statistics. Call while `.grad` is still live.

        Under HF that is `on_pre_optimizer_step` -- after clipping (so `grad_norm` carries
        the global clip factor) and before `model.zero_grad()`.
        """
        if not self._armed or not self._started:
            return
        cfg = self.cfg
        scalars: Dict[str, torch.Tensor] = {}
        total_sq: Optional[torch.Tensor] = None
        if cfg.include_total:
            for p in self.model.parameters():
                if p.grad is None or not p.requires_grad:
                    continue
                s = p.grad.detach().to(torch.float32).pow(2).sum()
                total_sq = s if total_sq is None else total_sq + s

        for name, mod in self.watched:
            params = [p for p in mod.parameters() if p.requires_grad]
            if cfg.gradients:
                grad_sq: Optional[torch.Tensor] = None
                zero_n: Optional[torch.Tensor] = None
                n_el = 0
                for p in params:
                    n_el += p.numel()
                    if p.grad is None:
                        continue
                    g = p.grad.detach().to(torch.float32)
                    s = g.pow(2).sum()
                    grad_sq = s if grad_sq is None else grad_sq + s
                    if cfg.grad_zero_frac:
                        z = (g == 0).sum()
                        zero_n = z if zero_n is None else zero_n + z
                if params:
                    dev = params[0].device
                    gs = torch.zeros((), device=dev) if grad_sq is None else grad_sq
                    scalars[f"{name}/grad_norm"] = gs.sqrt()
                    if total_sq is not None:
                        scalars[f"{name}/grad_frac"] = gs.sqrt() / (
                            total_sq.sqrt() + cfg.eps
                        )
                    if cfg.grad_zero_frac and n_el:
                        missing = float(n_el - sum(p.numel() for p in params if p.grad is not None))
                        zn = torch.zeros((), device=dev) if zero_n is None else zero_n.float()
                        scalars[f"{name}/grad_zero_frac"] = (zn + missing) / float(n_el)

            if cfg.weights and params:
                w_sq: Optional[torch.Tensor] = None
                for p in params:
                    s = p.detach().to(torch.float32).pow(2).sum()
                    w_sq = s if w_sq is None else w_sq + s
                if w_sq is not None:
                    scalars[f"{name}/weight_norm"] = w_sq.sqrt()
                    if f"{name}/grad_norm" in scalars:
                        scalars[f"{name}/update_ratio"] = scalars[f"{name}/grad_norm"] / (
                            w_sq.sqrt() + cfg.eps
                        )

            if cfg.weight_delta and name in self._init:
                snap = self._init[name]
                d_sq: Optional[torch.Tensor] = None
                for k, p in mod.named_parameters():
                    ref = snap.get(k)
                    if ref is None or ref.shape != p.shape:
                        continue
                    s = (p.detach().to(torch.float32) - ref.to(p.device)).pow(2).sum()
                    d_sq = s if d_sq is None else d_sq + s
                if d_sq is not None:
                    scalars[f"{name}/weight_delta"] = d_sq.sqrt()
                    scalars[f"{name}/weight_delta_rel"] = d_sq.sqrt() / (
                        self._init_norm.get(name, 0.0) + cfg.eps
                    )

        if total_sq is not None:
            scalars["_total/grad_norm"] = total_sq.sqrt()
        if self._embed is not None and self._embed_rows is not None:
            rows = self._embed.weight.detach()
            idx = self._embed_rows.to(rows.device)
            scalars["_ref/embed_norm"] = _row_norms(rows.index_select(0, idx)).mean()

        self._merge(scalars)

    def _merge(self, scalars: Dict[str, torch.Tensor]) -> None:
        """One host sync per device for every scalar collected this step, not one each."""
        if not scalars:
            return
        by_device: Dict[Any, List[str]] = {}
        for k, v in scalars.items():
            by_device.setdefault(v.device, []).append(k)
        for _, group in by_device.items():
            stacked = torch.stack([scalars[k].detach().reshape(()).float() for k in group])
            self._pending.update(dict(zip(group, stacked.cpu().tolist())))

    # -- drain -------------------------------------------------------------------------
    def drain(self) -> Dict[str, float]:
        """The metrics collected since the last drain, namespaced and flattened.

        Returns `{}` on an un-armed step, which is what makes throttling a true no-op at
        the logging layer rather than a filter applied after the work was already done.
        """
        if not self._started:
            return {}
        self._drain_buckets()
        if not self._pending:
            return {}
        out = dict(self._pending)
        self._pending.clear()

        embed = out.pop("_ref/embed_norm", None)
        if embed is not None:
            out["_ref/embed_norm"] = embed
            for name, _ in self.watched:
                act = out.get(f"{name}/act_norm_mean")
                if act is not None and embed > 0:
                    out[f"{name}/act_over_embed"] = act / embed

        out.update(self._static)
        out["_total/n_watched"] = float(len(self.watched))
        p = self.cfg.prefix
        return {f"{p}/{k}": float(v) for k, v in out.items()}

    def _drain_buckets(self) -> None:
        """Reduce the hook accumulators. Also the only place their buffers are released."""
        raw: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        for name, bucket in self._buckets.items():
            for key, tensor in bucket.sums.items():
                raw[f"{name}::{key}"] = tensor
            for key, n in bucket.counts.items():
                counts[f"{name}::{key}"] = n
            bucket.clear()
        if not raw:
            return
        keys = list(raw)
        # Grouped by device before stacking so this stays one transfer per device, not one
        # per scalar -- the hooks ran on whatever device the module lives on.
        values: Dict[str, float] = {}
        by_device: Dict[Any, List[str]] = {}
        for k in keys:
            by_device.setdefault(raw[k].device, []).append(k)
        for dev, group in by_device.items():
            stacked = torch.stack([raw[k].detach().reshape(()).float() for k in group])
            values.update(dict(zip(group, stacked.cpu().tolist())))

        def ratio(name: str, key: str, out: str) -> None:
            n = counts.get(f"{name}::{key}", 0)
            if n:
                self._pending[f"{name}/{out}"] = values[f"{name}::{key}"] / n

        for name, _ in self.watched:
            n_act = counts.get(f"{name}::act_sum", 0)
            if n_act:
                self._pending[f"{name}/act_norm_mean"] = values[f"{name}::act_sum"] / n_act
                self._pending[f"{name}/act_norm_min"] = values.get(f"{name}::act_min", 0.0)
                self._pending[f"{name}/act_norm_max"] = values.get(f"{name}::act_max", 0.0)
            ratio(name, "cap_sat_sum", "cap_sat_frac")
            ratio(name, "nl_sat_sum", "nonlin_sat_frac")
            ratio(name, "nl_dead_sum", "nonlin_dead_frac")

    # -- reporting ---------------------------------------------------------------------
    def describe(self) -> str:
        if not self.cfg.enabled:
            return "model metrics: disabled"
        rows = [
            f"model metrics: {len(self.watched)} module(s), every {self.cfg.every} step(s)"
            f" (plus the first {self.cfg.first_n})"
        ]
        for name, mod in self.watched:
            n = sum(p.numel() for p in mod.parameters())
            t = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            snap = "snapshot" if name in self._init else "no-snapshot"
            rows.append(f"  {name:<44} {type(mod).__name__:<22} {t}/{n} trainable  {snap}")
        return "\n".join(rows)


# ---------------------------------------------------------------------------------------
# HF Trainer integration
# ---------------------------------------------------------------------------------------
class ModelMetricsCallback:
    """Drives a `ModelMetrics` from `Trainer`'s callback points.

    Not a `TrainerCallback` subclass by inheritance-at-import: the class is created lazily
    in `attach_model_metrics` so this module stays importable (and testable) in an
    environment with no `transformers`. The methods below are the callback protocol.

    Any exception in here disables the collector and lets training continue. A metrics
    bug must never be able to kill a run that is otherwise fine -- the whole point is that
    it is cheap to leave on.
    """

    def __init__(self, metrics: ModelMetrics, verbose: bool = True):
        self.metrics = metrics
        self.verbose = verbose
        self.failed = False

    # -- protocol ----------------------------------------------------------------------
    def on_train_begin(self, args, state, control, **kwargs):
        self._guard(self._begin, args, state, kwargs)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        # `global_step` is the count of *completed* optimizer steps, so the step that is
        # starting now is the next one. Arming here covers every micro-batch of it.
        self._guard(lambda: self.metrics.arm(
            self.metrics.should_collect(int(state.global_step) + 1)
        ))
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self._guard(self.metrics.collect_gradients)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self._guard(lambda: self.metrics.arm(False))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._guard(self.metrics.stop)
        return control

    # -- glue --------------------------------------------------------------------------
    def _begin(self, args=None, state=None, kwargs=None):
        self.metrics.start()
        if self.verbose and getattr(args, "local_rank", -1) in (-1, 0):
            print(self.metrics.describe(), flush=True)

    def _guard(self, fn, *a):
        if self.failed or not self.metrics.cfg.enabled:
            return
        try:
            fn(*a) if a else fn()
        except Exception as exc:  # noqa: BLE001 - observability must not kill a run
            self.failed = True
            print(f"[model-metrics] disabled after an error: {exc!r}", flush=True)
            try:
                self.metrics.stop()
            except Exception:  # noqa: BLE001
                pass

    def drain(self) -> Dict[str, float]:
        if self.failed:
            return {}
        try:
            return self.metrics.drain()
        except Exception as exc:  # noqa: BLE001
            self.failed = True
            print(f"[model-metrics] disabled after an error: {exc!r}", flush=True)
            return {}


def _callback_class():
    """`ModelMetricsCallback` re-based on `TrainerCallback`, built on first use."""
    from transformers import TrainerCallback

    return type("ModelMetricsTrainerCallback", (ModelMetricsCallback, TrainerCallback), {})


def attach_model_metrics(
    trainer,
    config: Optional[ModelMetricsConfig] = None,
    args=None,
    verbose: bool = True,
):
    """Wire per-module metrics into an existing `Trainer`. Returns the callback, or None.

    Two touch points, both additive:

      * a `TrainerCallback` that arms the hooks and reads `.grad` at the right moments;
      * `trainer.log` wrapped so the collected scalars ride the run's existing logging
        path. Wrapping rather than mutating `logs` inside `on_log` is deliberate: the
        metrics then land in `state.log_history` (hence `trainer_state.json`) as well as
        in W&B, and the result does not depend on this callback happening to be ordered
        before `WandbCallback` in the handler.

    Called twice on the same trainer it is a no-op after the first.
    """
    cfg = config or (config_from_args(args) if args is not None else ModelMetricsConfig())
    if not cfg.enabled:
        return None
    if getattr(trainer, "_model_metrics", None) is not None:
        return trainer._model_metrics

    model = getattr(trainer, "model", None)
    if model is None:
        return None
    cb = _callback_class()(ModelMetrics(model, cfg), verbose=verbose)
    trainer.add_callback(cb)
    trainer._model_metrics = cb

    original_log = trainer.log

    def log(logs, start_time=None, _orig=original_log, _cb=cb):
        # Eval passes its own metrics dict through the same method; these are train-time
        # observations and would be mislabelled sitting next to `eval_*` keys.
        if not any(k.startswith("eval") for k in logs):
            extra = _cb.drain()
            if extra:
                logs.update(extra)
        return _orig(logs, start_time)

    trainer.log = log
    return cb


# ---------------------------------------------------------------------------------------
# argparse surface
# ---------------------------------------------------------------------------------------
def add_model_metrics_args(parser, group_title: str = "model metrics"):
    """The four flags every entry point gets. Defaults are the always-on setting.

    Kept to four on purpose: a diagnostic nobody can remember how to switch on is a
    diagnostic nobody switches on. Anything finer is reachable by constructing a
    `ModelMetricsConfig` and passing it to `attach_model_metrics`.
    """
    g = parser.add_argument_group(group_title)
    g.add_argument("--no-model-metrics", action="store_true",
                   help="disable per-module metrics entirely (no hooks, no snapshot, no "
                        "logged keys); the default is on and observational")
    g.add_argument("--model-metrics-every", type=int, default=25,
                   help="optimizer steps between collections; the first few steps are "
                        "always collected so a module dead at init shows up immediately")
    g.add_argument("--watch-modules", default=None,
                   help="comma-separated dotted module names or globs to watch, e.g. "
                        "'head,modality_embedder.encoders.*'. Omitted -> auto-discovery: "
                        "the root's non-backbone children plus their ModuleDict entries")
    g.add_argument("--model-metrics-exclude", default=None,
                   help="comma-separated names/globs removed from the watch set")
    return g


def config_from_args(args, **overrides) -> ModelMetricsConfig:
    """`ModelMetricsConfig` from a parsed namespace; unknown/absent flags fall back."""
    def _csv(text):
        if not text:
            return ()
        return tuple(x.strip() for x in str(text).split(",") if x.strip())

    cfg = ModelMetricsConfig(
        enabled=not bool(getattr(args, "no_model_metrics", False)),
        every=int(getattr(args, "model_metrics_every", 25) or 25),
        watch=_csv(getattr(args, "watch_modules", None)),
        exclude=_csv(getattr(args, "model_metrics_exclude", None)),
    )
    return replace(cfg, **overrides) if overrides else cfg
