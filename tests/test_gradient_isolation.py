"""Gradient-isolation pins for the state-probe co-trainer's row-type rules.

The invariant under audit (2026-08-23 follow-up): inappropriate row types must
never move weights they are not entitled to move. Specifically:

  I1  probe-only rows (ordered AND shuffled): the flow/action path (surrogate for
      readout head + velocity field) must receive EXACTLY-ZERO, POPULATED grads --
      zero because action_weight multiplies the flow loss by 0, populated (not None)
      because DDP with find_unused_parameters=False needs every param in the graph.
  I2  shuffled rows: value targets are all-NaN -> the value head's grads must be
      exactly zero (and populated), while distance/stop still train.
  I3  all-NaN probe targets (mask all-False): zero loss, zero grads, NO division by
      zero, NO NaN anywhere.
  I4  partial NaN coverage (pointnav ~80%): finite loss, finite nonzero grads.
  I5  the collator's zero-substitution of NaN action chunks is LOAD-BEARING:
      without it, 0.0 * NaN = NaN poisons every gradient in the step.
  I6  grad_scale scales ONLY the backbone path of the probe losses; heads get full
      gradient; grad_scale=0 detaches the backbone from the probe entirely.
  I7  mixed accumulation (demo row + probe-only row in one optimizer step): the
      action-path grads are BITWISE identical to the demo-only grads.
  I8  a head whose target COLUMN is absent (demo rows x stop head today) is not
      forwarded at all -> its params get grad=None. That is the documented DDP
      divergence hazard; pinned here as fact, not fixed here.
  I9  AdamW's decoupled weight decay updates a parameter whose grad is a populated
      zero tensor (and skips one whose grad is None) -- so an ALL-probe-only run
      still shrinks action-head weights by lr*wd per step. Quantified.

Standalone: imports the production `longnav.utils.state_probe` and the production
`ProbeCollator` (via the same module-exec technique as test_probe_trainer_units),
and NO retired script. The full backbone forward is NOT instantiated (CPU test);
what that does and does not cover is stated in the audit report.
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "src"), str(_ROOT / "data_scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from longnav.utils.state_probe import StateProbe, StateProbeConfig  # noqa: E402

HID = 16
T = 6


def tiny_probe(grad_scale=0.3):
    torch.manual_seed(1)
    cfg = StateProbeConfig(
        readout_offset=-2, grad_scale=grad_scale,
        distance={"hidden_dims": [8], "n_bins": 8, "d_max": 40.0,
                  "hl_sigma_ratio": 0.75, "loss_weight": 1.0},
        value={"hidden_dims": [8], "n_bins": 5, "v_min": -8.0, "v_max": 24.0,
               "hl_sigma_ratio": 0.75, "loss_weight": 1.0, "gamma": 0.97},
        stop={"hidden_dims": [4], "pos_weight": 15.0, "radius_m": 1.0,
              "loss_weight": 1.0},
    )
    return StateProbe(HID, cfg)


class Surrogate(nn.Module):
    """Mirrors train_flow_matching_sft_value.py's loss composition EXACTLY in shape:

        loss = flow_loss(targets) * action_weight + sum(probe.losses(hidden, ...))

    (forward:241-254 applies the multiplier; :317-318 adds the probe losses.)
    `backbone` stands in for backbone+LoRA (shared by both paths), `flow_head`
    for the readout MLP + velocity field (action path only).
    """

    def __init__(self, probe, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.backbone = nn.Linear(4, HID)
        self.flow_head = nn.Sequential(nn.Linear(HID, 8), nn.Tanh(), nn.Linear(8, 3))
        for m in [self.backbone, *self.flow_head]:
            if isinstance(m, nn.Linear):
                with torch.no_grad():
                    m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.3)
                    m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.1)
        self.probe = probe

    def forward(self, x, targets, action_weight, d=None, r=None, s=None):
        h = self.backbone(x)                       # (T, HID)
        pred = self.flow_head(h)                   # (T, 3)
        flow_loss = ((pred - targets) ** 2).mean()
        loss = flow_loss * action_weight
        assert torch.isfinite(loss), "scaled action loss must be finite (trainer:254)"
        losses = self.probe.losses(
            h.unsqueeze(0),
            None if d is None else d.reshape(1, -1),
            None if r is None else r.reshape(1, -1),
            stop_targets=None if s is None else s.reshape(1, -1),
        )
        for v in losses.values():
            loss = loss + v
        return loss, flow_loss


def groups(model):
    return {
        "backbone": list(model.backbone.parameters()),
        "flow_head": list(model.flow_head.parameters()),
        "dist_head": list(model.probe.distance_head.parameters()),
        "value_head": list(model.probe.value_head.parameters()),
        "stop_head": list(model.probe.stop_head.parameters()),
    }


def grad_state(params):
    """-> ('none' | 'zero' | 'nonzero' | 'nonfinite') over a parameter group."""
    gs = [p.grad for p in params]
    if all(g is None for g in gs):
        return "none"
    if any(g is None for g in gs):
        return "mixed-none"
    if any(not torch.isfinite(g).all() for g in gs):
        return "nonfinite"
    if all(bool((g == 0).all()) for g in gs):
        return "zero"
    return "nonzero"


def batch(kind):
    """Row fixtures. Distances/stops per-frame; returns NaN on shuffled rows."""
    torch.manual_seed(7)
    x = torch.randn(T, 4)
    if kind == "demo":
        return dict(x=x, targets=torch.randn(T, 3) * 0.1, action_weight=torch.tensor(1.0),
                    d=torch.rand(T) * 10, r=torch.rand(T) * 5, s=None)
    if kind == "probe_ordered":       # collator has zero-substituted the NaN chunk
        return dict(x=x, targets=torch.zeros(T, 3), action_weight=torch.tensor(0.0),
                    d=torch.rand(T) * 10, r=torch.rand(T) * 5,
                    s=(torch.rand(T) < 0.3).float())
    if kind == "probe_shuffled":      # value targets NaN BY DESIGN (builder:129-130)
        return dict(x=x, targets=torch.zeros(T, 3), action_weight=torch.tensor(0.0),
                    d=torch.rand(T) * 10, r=torch.full((T,), float("nan")),
                    s=(torch.rand(T) < 0.3).float())
    raise ValueError(kind)


def run(model, b):
    model.zero_grad(set_to_none=True)
    loss, flow_loss = model(b["x"], b["targets"], b["action_weight"],
                            d=b["d"], r=b["r"], s=b["s"])
    loss.backward()
    return loss, flow_loss


# ---------------------------------------------------------------------- I-tests


def test_I1_probe_only_rows_leave_action_path_populated_zero():
    m = Surrogate(tiny_probe())
    for kind in ("probe_ordered", "probe_shuffled"):
        loss, _ = run(m, batch(kind))
        g = {k: grad_state(v) for k, v in groups(m).items()}
        assert torch.isfinite(loss)
        assert g["flow_head"] == "zero", f"{kind}: action path must get populated zero grads, got {g['flow_head']}"
        assert g["backbone"] == "nonzero", f"{kind}: probe path (grad_scale) must reach backbone"
        assert g["dist_head"] == "nonzero"
        assert g["stop_head"] == "nonzero"


def test_I2_shuffled_row_value_head_exactly_zero_but_in_graph():
    m = Surrogate(tiny_probe())
    run(m, batch("probe_shuffled"))
    g = {k: grad_state(v) for k, v in groups(m).items()}
    assert g["value_head"] == "zero", \
        "NaN value targets must yield populated exactly-zero value-head grads (DDP-safe no-op)"
    assert g["dist_head"] == "nonzero" and g["stop_head"] == "nonzero"


def test_I3_all_false_mask_no_div0_no_nan():
    m = Surrogate(tiny_probe())
    b = batch("probe_shuffled")
    b["d"] = torch.full((T,), float("nan"))
    b["s"] = torch.full((T,), float("nan"))
    loss, _ = run(m, b)
    assert float(loss) == 0.0
    g = {k: grad_state(v) for k, v in groups(m).items()}
    assert g == {"backbone": "zero", "flow_head": "zero", "dist_head": "zero",
                 "value_head": "zero", "stop_head": "zero"}, g


def test_I4_partial_nan_coverage_is_finite_and_trains():
    m = Surrogate(tiny_probe())
    b = batch("demo")
    d = b["d"].clone()
    d[::3] = float("nan")            # ~1/3 sidecar gaps
    b["d"] = d
    loss, _ = run(m, b)
    assert torch.isfinite(loss)
    g = {k: grad_state(v) for k, v in groups(m).items()}
    assert g["dist_head"] == "nonzero" and g["backbone"] == "nonzero"
    # stop head is legitimately absent from the graph (demo rows carry no stop
    # column -- that is I8); what must NOT appear anywhere is a nonfinite grad.
    assert "nonfinite" not in g.values() and "mixed-none" not in g.values(), g


def test_I5_nan_poisoning_counterfactual():
    """0.0 * NaN = NaN: without the collator's zero-substitution the step dies.

    The production tripwire (`assert torch.isfinite(out['loss'])`, trainer:254)
    fires; here we show WHY -- the NaN reaches every gradient, not just the row's.
    """
    m = Surrogate(tiny_probe())
    b = batch("probe_ordered")
    b["targets"] = torch.full((T, 3), float("nan"))   # what the builder writes
    m.zero_grad(set_to_none=True)
    h = m.backbone(b["x"])
    flow_loss = ((m.flow_head(h) - b["targets"]) ** 2).mean()
    loss = flow_loss * b["action_weight"]             # 0.0 * NaN
    assert torch.isnan(loss), "0*NaN must be NaN -- zero-substitution is load-bearing"
    (loss + m.probe.losses(h.unsqueeze(0), b["d"].reshape(1, -1),
                           None, stop_targets=None)["probe/distance_loss"]).backward()
    assert grad_state(list(m.backbone.parameters())) == "nonfinite", \
        "the NaN poisons even the probe path's gradients through the shared backbone"


def test_I6_grad_scale_scales_backbone_path_only():
    b = batch("probe_ordered")
    grads = {}
    heads = {}
    for gs in (1.0, 0.25, 0.0):
        m = Surrogate(tiny_probe(grad_scale=gs), seed=0)
        run(m, b)
        grads[gs] = m.backbone.weight.grad.clone()
        heads[gs] = torch.cat([p.grad.flatten() for p in m.probe.distance_head.parameters()])
    assert torch.allclose(grads[0.25], grads[1.0] * 0.25, atol=1e-7), \
        "backbone grad must scale linearly with grad_scale"
    assert bool((grads[0.0] == 0).all()), "grad_scale=0 must fully detach the backbone"
    assert torch.allclose(heads[0.25], heads[1.0], atol=1e-7), \
        "head grads must be independent of grad_scale (straight-through)"
    assert heads[0.0].abs().sum() > 0


def test_I7_mixed_accumulation_probe_rows_add_bitwise_nothing_to_action_path():
    m = Surrogate(tiny_probe())
    run(m, batch("demo"))
    demo_only = [p.grad.clone() for p in m.flow_head.parameters()]
    probe_before = [p.grad.clone() for p in m.probe.distance_head.parameters()]

    # accumulate: demo then both probe-only kinds, no zero_grad between
    m.zero_grad(set_to_none=True)
    l, _ = m(**batch("demo")); l.backward()
    l, _ = m(**batch("probe_ordered")); l.backward()
    l, _ = m(**batch("probe_shuffled")); l.backward()
    for p, g0 in zip(m.flow_head.parameters(), demo_only):
        assert torch.equal(p.grad, g0), \
            "probe-only rows must contribute BITWISE zero to the action path"
    changed = any(not torch.equal(p.grad, g0) for p, g0 in
                  zip(m.probe.distance_head.parameters(), probe_before))
    assert changed, "probe rows must still train the probe heads in the same step"
    for group, ps in groups(m).items():
        assert grad_state(ps) in ("zero", "nonzero"), f"{group} grads must stay finite"


def test_I8_absent_stop_column_leaves_stop_head_out_of_graph():
    """Demo corpora carry no stop_targets -> stop head never forwards -> grad None.
    This is the DDP divergence hazard for a future mixed stop-head run; pinned as
    the CURRENT behavior so the fix (NaN columns or a zero_touch) is measurable."""
    m = Surrogate(tiny_probe())
    b = batch("demo")
    b["s"] = None
    run(m, b)
    assert grad_state(list(m.probe.stop_head.parameters())) == "none"


def test_I9_adamw_weight_decay_moves_zero_grad_params():
    lr, wd = 1e-4, 1e-4
    p_zero = nn.Parameter(torch.ones(4))
    p_none = nn.Parameter(torch.ones(4))
    opt = torch.optim.AdamW([p_zero, p_none], lr=lr, weight_decay=wd)
    p_zero.grad = torch.zeros_like(p_zero)     # populated zero (probe-only row)
    p_none.grad = None
    opt.step()
    assert torch.allclose(p_zero, torch.full((4,), 1 - lr * wd)), \
        "decoupled decay applies to zero-grad params: all-probe-only runs DO shrink the action head"
    assert torch.equal(p_none, torch.ones(4)), "grad=None params are skipped entirely"


# ------------------------------------------------- collator zero-substitution


@pytest.fixture(scope="module")
def tfsv():
    spec = importlib.util.spec_from_file_location(
        "tfsv_gradiso", str(_ROOT / "data_scripts" / "train_flow_matching_sft_value.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tfsv_gradiso"] = mod
    spec.loader.exec_module(mod)
    return mod


def _collator(tfsv, monkeypatch, n=4):
    c = object.__new__(tfsv.ProbeCollator)
    c._rng = np.random.default_rng(3)
    monkeypatch.setattr(
        tfsv._BaseCollator, "__call__",
        lambda self, examples: {"targets": torch.tensor(
            examples[0]["action_chunks"], dtype=torch.float32)})
    return c


def test_collator_zero_substitutes_nan_chunks_and_zeroes_weight(tfsv, monkeypatch):
    n = 4
    col = _collator(tfsv, monkeypatch, n)
    row = {"action_chunks": [[float("nan")] * 3] * n, "probe_only": True,
           "distance_targets": [1.0] * n, "return_targets": [None] * n,
           "stop_targets": [0.0] * n}
    out = col([row])
    assert bool((out["targets"] == 0).all()), "NaN chunks must be zero-substituted"
    assert float(out["action_weight"]) == 0.0
    assert out["probe_only"] is True
    assert torch.isnan(out["return_targets"]).all(), "NaN value targets must SURVIVE (mask, not zero)"
    assert torch.isfinite(out["distance_targets"]).all()


def test_collator_refuses_probe_only_row_with_finite_actions(tfsv, monkeypatch):
    col = _collator(tfsv, monkeypatch)
    row = {"action_chunks": [[0.1, 0.0, 0.0]] * 4, "probe_only": True,
           "distance_targets": [1.0] * 4}
    with pytest.raises(ValueError, match="finite action targets"):
        col([row])


def test_collator_demo_row_keeps_weight_one(tfsv, monkeypatch):
    col = _collator(tfsv, monkeypatch)
    row = {"action_chunks": [[0.1, 0.2, 0.0]] * 4, "distance_targets": [1.0] * 4,
           "return_targets": [2.0] * 4}
    out = col([row])
    assert float(out["action_weight"]) == 1.0
    assert torch.isfinite(out["targets"]).all()
