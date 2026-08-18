"""Unit pins for the derived trainer's collator logic (no model, no processor).

The full path (real collator + backbone + probe) is covered by the GPU smoke
(tests/smoke_probe_trainer.sh); what is pinned here is the window-start recovery
and slice arithmetic, because an off-by-one there silently trains the probe on
the wrong frames' labels.
"""

import importlib.util
import sys

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "tfsv", "data_scripts/train_flow_matching_sft_value.py")
tfsv = importlib.util.module_from_spec(spec)
sys.modules["tfsv"] = tfsv
spec.loader.exec_module(tfsv)


class _StubBase:
    """Stands in for TurnVectorCollator.__call__: draws the rng like the real one."""

    def __init__(self, n_total, cap, train):
        self.n_total, self.cap, self.train = n_total, cap, train

    def call(self, collator, examples):
        start = 0
        n_kept = self.n_total
        if self.cap is not None and self.n_total > self.cap:
            start = int(collator._rng.integers(0, self.n_total - self.cap + 1)) \
                if self.train else 0
            n_kept = self.cap
        return {"targets": torch.zeros(n_kept, 3)}


def make_collator(n_total, cap, train, monkeypatch, seed=7):
    import numpy as np
    c = object.__new__(tfsv.ProbeCollator)
    c._rng = np.random.default_rng(seed)
    stub = _StubBase(n_total, cap, train)
    monkeypatch.setattr(tfsv._BaseCollator, "__call__",
                        lambda self, examples: stub.call(self, examples))
    return c


def test_windowed_slice_matches_recovered_start(monkeypatch):
    n, cap = 12, 5
    col = make_collator(n, cap, train=True, monkeypatch=monkeypatch)
    ex = {"distance_targets": [float(i) for i in range(n)],
          "return_targets": [float(10 + i) for i in range(n)]}
    out = col([ex])
    # recover what start must have been from the emitted values themselves
    start = int(out["distance_targets"][0].item())
    assert 0 <= start <= n - cap
    assert out["distance_targets"].tolist() == [float(start + j) for j in range(cap)]
    assert out["return_targets"].tolist() == [float(10 + start + j) for j in range(cap)]


def test_unwindowed_uses_full_columns(monkeypatch):
    n = 4
    col = make_collator(n, cap=None, train=True, monkeypatch=monkeypatch)
    ex = {"distance_targets": [1.0, 2.0, None, 4.0], "return_targets": [0.0] * n}
    out = col([ex])
    assert out["distance_targets"].shape[0] == n
    assert torch.isnan(out["distance_targets"][2])   # None -> NaN, maskable


def test_missing_columns_are_simply_absent(monkeypatch):
    col = make_collator(6, cap=None, train=True, monkeypatch=monkeypatch)
    out = col([{}])
    assert "distance_targets" not in out and "return_targets" not in out


def test_short_column_is_refused(monkeypatch):
    col = make_collator(8, cap=5, train=True, monkeypatch=monkeypatch)
    with pytest.raises(ValueError, match="misaligned"):
        col([{"distance_targets": [1.0, 2.0]}])


def test_multi_draw_base_is_refused(monkeypatch):
    col = make_collator(8, cap=5, train=True, monkeypatch=monkeypatch)

    def double_draw(self, examples):
        self._rng.integers(0, 3)
        self._rng.integers(0, 3)
        return {"targets": torch.zeros(5, 3)}
    monkeypatch.setattr(tfsv._BaseCollator, "__call__", double_draw)
    with pytest.raises(AssertionError, match="exactly one draw"):
        col([{"distance_targets": [0.0] * 8}])


def test_no_state_probe_class_attribute_shadow():
    """nn.Module stores assigned submodules in `_modules`, reachable only via
    __getattr__ -- which never fires if a CLASS attribute satisfies lookup first.
    A `state_probe = None` class default therefore permanently shadows the real
    probe (observed: silent zero-grad training). Pin its absence."""
    import torch.nn as nn
    cls = tfsv.TurnFlowActionRegressorWithProbe
    assert "state_probe" not in vars(cls), \
        "state_probe class attribute reintroduced; it shadows the registered submodule"

    class Dummy(nn.Module):
        pass

    d = Dummy()
    probe = nn.Linear(2, 2)
    d.state_probe = probe                       # registers as submodule
    assert getattr(d, "state_probe", None) is probe
    assert "state_probe" in d._modules          # and it IS module-registered
