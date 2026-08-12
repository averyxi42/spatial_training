"""The `stop_*` / `creep_*` / `near_zero_*` band metrics are off by default, not gone.

They exist to compare a discrete head's codebook occupancy against a continuous head's --
the near-zero mass is expressible by both, where `stop_*` alone is not (a codebook with a
centroid exactly on zero clears the EXACT edge trivially). That is a real question, but not
one the current data mixture asks, and the breakdown is ~30 keys on every log line, which
crowds out `turn_loss` / `rmse_*` / `pose_rmse_*`.

So: gated, with the sums still accumulated either way. Turning the flag on must recover
exactly what was logged before, and turning it off must not disturb anything else.
"""

import pytest
import torch

from longnav.utils.flow_matching_head import FlowMatchingSFTTrainer

PARENT = FlowMatchingSFTTrainer.__mro__[1]

SUMS = {
    "n_rows": torch.tensor(4.0),
    "sum_stop_pred": torch.tensor([1.0, 1.0, 1.0]),
    "sum_stop_gt": torch.tensor([1.0, 1.0, 1.0]),
    "sum_creep_pred": torch.tensor([1.0, 1.0, 1.0]),
    "sum_creep_gt": torch.tensor([2.0, 2.0, 2.0]),
}


def drain(emit, monkeypatch):
    """Run the real `_drain_metrics` with the parent's contribution stubbed."""
    monkeypatch.setattr(
        PARENT, "_drain_metrics", lambda self, prefix="": {f"{prefix}turn_loss": 0.1}
    )
    inst = FlowMatchingSFTTrainer.__new__(FlowMatchingSFTTrainer)
    inst._sums = dict(SUMS)
    inst.emit_motion_bands = emit
    return inst._drain_metrics("")


def bands(out):
    return sorted(k for k in out if k.startswith(("stop_", "creep_", "near_zero_")))


def test_off_by_default():
    assert FlowMatchingSFTTrainer.emit_motion_bands is False


def test_disabled_emits_no_band_keys(monkeypatch):
    out = drain(False, monkeypatch)
    assert bands(out) == []
    assert "turn_loss" in out, "gating must not swallow the metrics people actually read"


def test_enabled_recovers_them(monkeypatch):
    out = drain(True, monkeypatch)
    got = bands(out)
    assert got, "the flag must bring the breakdown back"
    assert any(k.startswith("stop_pred_") for k in got)
    assert any(k.startswith("creep_gt_") for k in got)
    assert any(k.startswith("near_zero_") for k in got)


def test_the_values_are_unchanged_by_gating(monkeypatch):
    """Enabling must reproduce the old numbers, not a re-derived approximation."""
    out = drain(True, monkeypatch)
    # 1.0 summed over n_rows=4
    assert out["stop_pred_dx"] == pytest.approx(0.25)
    # near_zero = stop + creep, over rows
    assert out["near_zero_pred_dx"] == pytest.approx(0.5)
    assert out["near_zero_gt_dx"] == pytest.approx(0.75)
    assert out["near_zero_ratio_dx"] == pytest.approx(2.0 / 3.0)


def test_the_sums_are_still_accumulated_when_disabled(monkeypatch):
    """The gate is on emission only -- nothing downstream loses the ability to compute
    these, and flipping the flag needs no re-run of anything."""
    out = drain(False, monkeypatch)
    assert bands(out) == []
    again = drain(True, monkeypatch)
    assert again["near_zero_pred_dx"] == pytest.approx(0.5)
