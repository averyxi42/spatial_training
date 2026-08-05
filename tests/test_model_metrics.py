"""Tests for `longnav.utils.model_metrics`.

No GPU, no VLM, no `Trainer`: the collector's contract is `(module, gradients) ->
scalars`, and toy `nn.Module`s exercise all of it. The one test that reaches into the
project proper reconstructs the `zero_init + max_norm` condition that froze
`run_v6_regress_pose`'s pose encoder for ~3600 steps, and asserts this module would have
said so on the first collected step.

    pytest tests/test_model_metrics.py -q
"""

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
import torch
from torch import nn

from longnav.utils.model_metrics import (
    ModelMetrics,
    ModelMetricsConfig,
    auto_watch,
    resolve_watch,
)


# ---------------------------------------------------------------------------------------
# toys
# ---------------------------------------------------------------------------------------
class DeadBlock(nn.Module):
    """A block whose output is multiplied by exactly zero, so no gradient ever reaches it.

    The same shape of defect as the real one: the parameters are trainable, the module is
    in the optimizer, the forward runs -- and the backward delivers exactly nothing.
    """

    def __init__(self, d=8):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x) * 0.0


class LiveBlock(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.lin(x))


class SaturatedBlock(nn.Module):
    """`nn.Tanh` driven far past its knee -- the bounded-nonlinearity failure.

    The weights are zeroed and the bias set to the gain, so every unit is pinned at
    `tanh(50) == 1` for any input. Scaling random weights instead leaves a few units near
    the knee and turns an exact assertion into a threshold that drifts with the seed.
    """

    def __init__(self, d=8, gain=50.0):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.act = nn.Tanh()
        with torch.no_grad():
            self.lin.weight.zero_()
            self.lin.bias.fill_(gain)

    def forward(self, x):
        return self.act(self.lin(x))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.dead = DeadBlock()
        self.live = LiveBlock()
        self.sat = SaturatedBlock()
        self.blocks = nn.ModuleDict({"a": LiveBlock(), "b": DeadBlock()})

    def forward(self, x):
        y = self.dead(x) + self.live(x) + self.sat(x)
        for m in self.blocks.values():
            y = y + m(x)
        return y


def run_step(model, metrics, step, batch=4, dim=8):
    """One armed-or-not train step: arm, forward, backward, collect, drain.

    The input is seeded from the step so a test's numbers do not depend on how many other
    tests drew from the global RNG before it.
    """
    metrics.arm(metrics.should_collect(step))
    x = torch.randn(batch, dim, generator=torch.Generator().manual_seed(1000 + step))
    loss = model(x).pow(2).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    metrics.collect_gradients()
    out = metrics.drain()
    metrics.arm(False)
    return out


# ---------------------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------------------
def test_auto_watch_picks_children_and_expands_moduledicts():
    model = Toy()
    names = [n for n, _ in auto_watch(model, ModelMetricsConfig())]
    assert {"dead", "live", "sat", "blocks"} <= set(names)
    # A ModuleDict is expanded: one dead encoder among several must not be averaged away.
    assert "blocks.a" in names and "blocks.b" in names


def test_auto_watch_skips_the_backbone():
    class WithBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(600, 600)  # 360k params
            self.head = nn.Linear(8, 8)

    cfg = ModelMetricsConfig(auto_max_params=100_000)
    names = [n for n, _ in auto_watch(WithBackbone(), cfg)]
    assert names == ["head"]


def test_resolve_watch_globs_and_typos():
    model = Toy()
    names = [n for n, _ in resolve_watch(model, ["blocks.*", "live"])]
    assert "live" in names and "blocks.a" in names and "blocks.b" in names
    assert "dead" not in names
    with pytest.raises(KeyError):
        resolve_watch(model, ["haed"])  # an exact name that matches nothing is a typo


# ---------------------------------------------------------------------------------------
# the headline detections
# ---------------------------------------------------------------------------------------
def test_frozen_submodule_is_detected():
    model = Toy()
    mm = ModelMetrics(model, ModelMetricsConfig(every=1)).start()
    for step in range(1, 4):
        out = run_step(model, mm, step)
        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-0.1)

    assert out["model/dead/grad_norm"] == 0.0
    assert out["model/dead/grad_zero_frac"] == 1.0
    assert out["model/dead/grad_frac"] == 0.0
    # The weight-delta is the claim that cannot be argued with: it has not moved at all.
    assert out["model/dead/weight_delta"] == 0.0
    assert out["model/dead/weight_delta_rel"] == 0.0

    # ... and the live one has, so the test is not just asserting that nothing works.
    assert out["model/live/grad_norm"] > 0.0
    assert out["model/live/weight_delta"] > 0.0
    assert out["model/live/grad_frac"] > 0.0
    mm.stop()


def test_saturating_nonlinearity_is_reported():
    model = Toy()
    mm = ModelMetrics(model, ModelMetricsConfig(every=1)).start()
    out = run_step(model, mm, 1)
    assert out["model/sat/nonlin_sat_frac"] == pytest.approx(1.0)
    # The healthy block's ReLU is nowhere near all-dead.
    assert 0.0 <= out["model/live/nonlin_dead_frac"] < 0.95
    mm.stop()


def test_output_norm_pinned_at_a_max_norm_cap_is_reported():
    """A module carrying its own `max_norm` gets `cap_sat_frac` -- defect (2)."""

    class Capped(nn.Module):
        def __init__(self, cap=1.0):
            super().__init__()
            self.lin = nn.Linear(8, 8)
            self.max_norm = cap

        def forward(self, x):
            out = self.lin(x) * 1000.0
            norm = out.norm(dim=-1, keepdim=True)
            return out * torch.tanh(norm / self.max_norm) * self.max_norm / (norm + 1e-12)

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.capped = Capped()

        def forward(self, x):
            return self.capped(x)

    model = Root()
    mm = ModelMetrics(model, ModelMetricsConfig(every=1)).start()
    out = run_step(model, mm, 1)
    assert out["model/capped/cap_sat_frac"] == pytest.approx(1.0)
    assert out["model/capped/act_norm_mean"] == pytest.approx(1.0, abs=1e-3)
    mm.stop()


def test_activation_norm_is_measured_against_the_token_embedding():
    """Defect (3): an injected vector 11x an ordinary token embedding, unnoticed."""

    class Injector(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.lin = nn.Linear(4, 16)
            with torch.no_grad():
                self.lin.weight.mul_(scale)
                self.lin.bias.zero_()

        def forward(self, x):
            return self.lin(x)

    class Root(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.embed = nn.Embedding(512, 16)
            self.inject = Injector(scale)
            with torch.no_grad():
                self.embed.weight.normal_(0, 1.0 / math.sqrt(16))  # row norms ~1

        def forward(self, x):
            return self.inject(x).sum() + self.embed.weight.sum() * 0.0

    quiet = Root(1.0)
    mm = ModelMetrics(quiet, ModelMetricsConfig(every=1, embed_sample=512)).start()
    mm.arm(True)
    quiet(torch.randn(4, 4)).backward()
    mm.collect_gradients()
    quiet_out = mm.drain()
    mm.stop()

    loud = Root(20.0)
    mm = ModelMetrics(loud, ModelMetricsConfig(every=1, embed_sample=512)).start()
    mm.arm(True)
    loud(torch.randn(4, 4)).backward()
    mm.collect_gradients()
    loud_out = mm.drain()
    mm.stop()

    assert quiet_out["model/_ref/embed_norm"] == pytest.approx(1.0, abs=0.15)
    assert loud_out["model/inject/act_over_embed"] > 10 * quiet_out["model/inject/act_over_embed"]


# ---------------------------------------------------------------------------------------
# cost controls
# ---------------------------------------------------------------------------------------
def test_disabled_is_a_true_no_op():
    model = Toy()
    mm = ModelMetrics(model, ModelMetricsConfig(enabled=False))
    mm.start()
    assert mm.watched == []
    assert mm._handles == []  # no hooks were ever registered
    hooks = sum(
        len(m._forward_hooks) for m in model.modules()
    )
    assert hooks == 0
    for step in range(1, 5):
        assert run_step(model, mm, step) == {}
    mm.stop()


def test_throttling_collects_only_on_scheduled_steps():
    model = Toy()
    cfg = ModelMetricsConfig(every=10, first_n=2)
    mm = ModelMetrics(model, cfg).start()
    collected = [step for step in range(1, 31) if run_step(model, mm, step)]
    # The opening steps regardless, then every tenth.
    assert collected == [1, 2, 10, 20, 30]
    mm.stop()


def test_unarmed_hooks_do_no_work():
    """The hooks stay attached between collections; the guard is the boolean, not removal."""
    model = Toy()
    mm = ModelMetrics(model, ModelMetricsConfig(every=1000, first_n=0)).start()
    assert mm._handles, "hooks are registered up front"
    run_step(model, mm, 1)
    assert all(not b.sums and not b.counts for b in mm._buckets.values())
    mm.stop()


def test_stop_detaches_every_hook():
    model = Toy()
    mm = ModelMetrics(model, ModelMetricsConfig()).start()
    assert sum(len(m._forward_hooks) for m in model.modules()) > 0
    mm.stop()
    assert sum(len(m._forward_hooks) for m in model.modules()) == 0


def test_weight_delta_is_measured_from_start_not_construction():
    """`start()` is the datum, so a resumed run measures movement since the resume."""
    model = Toy()
    mm = ModelMetrics(model, ModelMetricsConfig(every=1))
    with torch.no_grad():  # stand-in for Trainer restoring a checkpoint
        model.live.lin.weight.add_(5.0)
    mm.start()
    out = run_step(model, mm, 1)
    assert out["model/live/weight_delta"] == 0.0
    with torch.no_grad():
        model.live.lin.weight.add_(1.0)
    out = run_step(model, mm, 2)
    assert out["model/live/weight_delta"] == pytest.approx(
        math.sqrt(model.live.lin.weight.numel()), rel=1e-4
    )
    mm.stop()


# ---------------------------------------------------------------------------------------
# regression: the exact defect that cost the GPU-weeks
# ---------------------------------------------------------------------------------------
def test_fourier_se2_zero_init_plus_max_norm_is_flagged():
    """`FourierSE2Encoder(zero_init=True, max_norm=m)` takes exactly zero gradient.

    The output layer starts at zero, so `|out| == 0`, and the soft cap multiplies by
    `tanh(|out|/m) * m / (|out| + 1e-12) == 0`. The backward carries the same factor, so
    the encoder can never leave the origin. `run_v6_regress_pose` trained ~3600 steps in
    this state. This asserts the collector says so on the first collected step -- and, as
    the control, that the same encoder without the cap is reported as learning.
    """
    modality_embed = pytest.importorskip("longnav.utils.modality_embed")
    FourierSE2Encoder = modality_embed.FourierSE2Encoder

    class Wrapper(nn.Module):
        def __init__(self, **kw):
            super().__init__()
            self.encoder = FourierSE2Encoder(n_features=3, d_model=16, **kw)

        def forward(self, pose):
            return self.encoder(pose)

    pose = torch.tensor([[1.0, 2.0, 0.5], [-3.0, 0.25, -2.0]])

    # `sum()`, not `pow(2).sum()`: a squared loss has zero gradient at a zero output for
    # trivial reasons, which would make the control below pass for the wrong reason. This
    # loss has a nonzero gradient w.r.t. the output, so anything that stays at zero does so
    # because of the cap.
    dead = Wrapper(zero_init=True, max_norm=8.0)
    mm = ModelMetrics(dead, ModelMetricsConfig(every=1)).start()
    mm.arm(True)
    out = dead(pose)
    out.sum().backward()
    mm.collect_gradients()
    metrics = mm.drain()
    mm.stop()

    assert float(out.detach().abs().max()) == 0.0, "premise: capped zero-init output is zero"
    assert metrics["model/encoder/grad_norm"] == 0.0
    assert metrics["model/encoder/grad_zero_frac"] == 1.0
    assert metrics["model/encoder/act_norm_mean"] == 0.0

    live = Wrapper(zero_init=True, max_norm=None)
    mm = ModelMetrics(live, ModelMetricsConfig(every=1)).start()
    mm.arm(True)
    live(pose).sum().backward()
    mm.collect_gradients()
    metrics = mm.drain()
    mm.stop()
    assert metrics["model/encoder/grad_norm"] > 0.0
    assert metrics["model/encoder/grad_zero_frac"] < 1.0


def test_auto_watch_on_the_real_regressor_layout():
    """The default watch set on `TurnVectorRegressor`'s actual attribute layout.

    Built from the real `ModalityEmbedder` with two specs, because the case that matters
    is two encoders where only one is dead -- an aggregate over `modality_embedder` would
    halve the signal instead of showing it.
    """
    modality_embed = pytest.importorskip("longnav.utils.modality_embed")

    embedder = modality_embed.ModalityEmbedder(
        [
            {"token": "<pose>", "n_features": 3, "encoder": "fourier_se2", "column": "p"},
            {"token": "<goal>", "n_features": 4, "encoder": "mlp", "column": "g"},
        ],
        d_model=16,
    )

    class Regressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(1000, 1000)  # stands in for the 2B VLM
            self.head = nn.Linear(16, 3)
            self.stop_head = nn.Linear(16, 1)
            self.modality_embedder = embedder

    # The cap sits between the encoders (~10^5 params each) and the stand-in backbone
    # (10^6), which is the same separation the real 50M default draws against a 2B VLM.
    cfg = ModelMetricsConfig(auto_max_params=500_000)
    names = [n for n, _ in auto_watch(Regressor(), cfg)]
    assert "backbone" not in names
    assert {"head", "stop_head", "modality_embedder"} <= set(names)
    assert "modality_embedder.encoders.pose" in names
    assert "modality_embedder.encoders.goal" in names


# ---------------------------------------------------------------------------------------
# Trainer wiring
# ---------------------------------------------------------------------------------------
def test_attaches_to_a_real_trainer_and_reaches_log_history(tmp_path):
    """The wiring, end to end: hooks armed at the right step, scalars in `log_history`.

    A CPU toy and four steps -- the point is the callback ordering (`.grad` must still be
    live when we read it) and the `trainer.log` wrapper, both of which are properties of
    HF's inner loop rather than of anything in this repo.
    """
    transformers = pytest.importorskip("transformers")
    from torch.utils.data import Dataset

    from longnav.utils.model_metrics import attach_model_metrics

    class Rows(Dataset):
        def __len__(self):
            return 32

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return {"x": torch.randn(8, generator=g)}

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.dead = DeadBlock()
            self.live = LiveBlock()

        def forward(self, x, **kw):
            return {"loss": (self.dead(x) + self.live(x)).pow(2).mean()}

    args = transformers.TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=4,
        per_device_train_batch_size=4,
        logging_steps=1,
        report_to=[],
        save_strategy="no",
        use_cpu=True,
        remove_unused_columns=False,
    )
    trainer = transformers.Trainer(model=Net(), args=args, train_dataset=Rows())
    cb = attach_model_metrics(
        trainer, config=ModelMetricsConfig(every=1000, first_n=2), verbose=False
    )
    assert cb is not None
    assert attach_model_metrics(trainer, verbose=False) is cb  # idempotent
    trainer.train()

    logged = [r for r in trainer.state.log_history if "model/dead/grad_norm" in r]
    steps = [r["step"] for r in logged]
    assert steps == [1, 2], f"collected on the wrong steps: {steps}"
    assert all(r["model/dead/grad_norm"] == 0.0 for r in logged)
    assert all(r["model/dead/weight_delta"] == 0.0 for r in logged)
    assert logged[-1]["model/live/grad_norm"] > 0.0
    assert logged[-1]["model/live/weight_delta"] > 0.0
    # Hooks are gone once training ends -- nothing survives to leak into a later phase.
    assert sum(len(m._forward_hooks) for m in trainer.model.modules()) == 0


def test_disabled_attach_leaves_the_trainer_untouched(tmp_path):
    transformers = pytest.importorskip("transformers")

    from longnav.utils.model_metrics import attach_model_metrics

    args = transformers.TrainingArguments(output_dir=str(tmp_path), report_to=[])
    trainer = transformers.Trainer(model=nn.Linear(4, 4), args=args)
    n_callbacks = len(trainer.callback_handler.callbacks)

    assert attach_model_metrics(trainer, ModelMetricsConfig(enabled=False)) is None
    assert len(trainer.callback_handler.callbacks) == n_callbacks
    # `trainer.log` is still the class's bound method, not an instance-level wrapper
    # (bound methods are fresh objects per access, so compare the function).
    assert "log" not in vars(trainer)
    assert trainer.log.__func__ is transformers.Trainer.log


# ---------------------------------------------------------------------------------------
# argparse / config surface
# ---------------------------------------------------------------------------------------
def test_config_from_args_round_trip():
    import argparse

    from longnav.utils.model_metrics import add_model_metrics_args, config_from_args

    p = argparse.ArgumentParser()
    add_model_metrics_args(p)

    cfg = config_from_args(p.parse_args([]))
    assert cfg.enabled and cfg.every == 25 and cfg.watch == ()

    cfg = config_from_args(p.parse_args(["--no-model-metrics"]))
    assert not cfg.enabled

    cfg = config_from_args(
        p.parse_args(["--model-metrics-every", "5",
                      "--watch-modules", "head, modality_embedder.encoders.*",
                      "--model-metrics-exclude", "head"])
    )
    assert cfg.every == 5
    assert cfg.watch == ("head", "modality_embedder.encoders.*")
    assert cfg.exclude == ("head",)
