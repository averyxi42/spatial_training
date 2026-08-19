"""Pins for longnav.utils.state_probe: return derivation, HL-Gauss bins in log space,
the P(d<=r) stop rule, NaN masking, and checkpoint IO."""

import math

import pytest
import torch

from longnav.utils.state_probe import (
    LogDistanceHead,
    StateProbe,
    StateProbeConfig,
    ValueDistHead,
    distance_return_targets,
    load_state_probe,
    save_state_probe,
)


class TestReturnTargets:
    def test_telescopes_at_gamma_one(self):
        d = [5.0, 4.0, 2.5, 2.0, 2.0]
        r = distance_return_targets(d, gamma=1.0, reward_clip=10.0)
        # unclipped, gamma=1: return-to-go telescopes to d_k - d_final
        for k in range(len(d)):
            assert r[k] == pytest.approx(d[k] - d[-1])

    def test_clip_binds(self):
        r = distance_return_targets([10.0, 1.0], gamma=0.97)
        assert r[0] == pytest.approx(0.75)   # 9 m step clipped to 0.75

    def test_nonfinite_transition_pays_zero(self):
        r = distance_return_targets([3.0, None, 1.0], gamma=0.5)
        assert r == [0.0, 0.0, 0.0]  # both transitions touch the None

    def test_gamma_discounts(self):
        d = [3.0, 2.5, 2.0, 2.0]
        r = distance_return_targets(d, gamma=0.5)
        assert r[0] == pytest.approx(0.5 + 0.5 * 0.5)
        assert r[-1] == 0.0


class TestLogDistanceHead:
    def head(self):
        torch.manual_seed(0)
        return LogDistanceHead(input_dim=8, hidden_dims=[16], n_bins=32, d_max=40.0)

    def test_bin_widths_grow_with_distance(self):
        h = self.head()
        w = torch.diff(h.edges_m)
        assert w[0] < 0.2 and w[-1] > 1.0 and bool((torch.diff(w) > 0).all())

    def test_targets_are_distributions_and_mask_nan(self):
        h = self.head()
        y = torch.tensor([0.3, 5.0, float("nan")])
        t = h.targets(torch.nan_to_num(y))
        assert torch.allclose(t.sum(-1), torch.ones(3), atol=1e-5)
        logits = torch.zeros(3, 32)
        loss = h.loss(logits, y)          # NaN row masked automatically
        assert torch.isfinite(loss)

    def test_p_within_is_calibrated_on_a_delta(self):
        h = self.head()
        # concentrate all mass in the bin containing d=0.5
        b = int(torch.searchsorted(h.edges_m, torch.tensor(0.5))) - 1
        logits = torch.full((1, 32), -30.0)
        logits[0, b] = 30.0
        assert float(h.p_within(logits, 40.0)) == pytest.approx(1.0, abs=1e-4)
        assert float(h.p_within(logits, 0.01)) == pytest.approx(0.0, abs=1e-4)
        lo, hi = float(h.edges_m[b]), float(h.edges_m[b + 1])
        mid = 0.5 * (lo + hi)
        assert float(h.p_within(logits, mid)) == pytest.approx(0.5, abs=0.02)

    def test_expectation_decodes_in_meters(self):
        h = self.head()
        b = 10
        logits = torch.full((1, 32), -30.0)
        logits[0, b] = 30.0
        assert float(h.expectation(logits)) == pytest.approx(float(h.centers_orig[b]), rel=1e-4)


class TestValueDistHead:
    def test_mean_preserved_in_range(self):
        torch.manual_seed(0)
        h = ValueDistHead(input_dim=8, hidden_dims=[], n_bins=41, v_min=-8.0, v_max=24.0)
        y = torch.tensor([0.0, 3.7, -2.2, 17.0])
        t = h.targets(y)
        dec = t @ h.centers_orig
        assert torch.allclose(dec, y, atol=0.15)   # within a fraction of a bin


class TestProbeAndIO:
    def test_losses_and_roundtrip(self, tmp_path):
        torch.manual_seed(0)
        cfg = StateProbeConfig()
        probe = StateProbe(input_dim=8, cfg=cfg)
        hidden = torch.randn(2, 5, 8)
        d = torch.rand(2, 5) * 10
        r = torch.rand(2, 5)
        mask = torch.ones(2, 5)
        losses = probe.losses(hidden, distance_targets=d, return_targets=r, mask=mask)
        assert set(losses) == {"probe/distance_loss", "probe/value_loss"}
        assert all(torch.isfinite(v) for v in losses.values())
        total = sum(losses.values())
        total.backward()
        save_state_probe(tmp_path, probe)
        probe2 = load_state_probe(tmp_path, input_dim=8)
        for (k1, v1), (k2, v2) in zip(sorted(probe.state_dict().items()),
                                      sorted(probe2.state_dict().items())):
            assert k1 == k2 and torch.equal(v1, v2)

    def test_grad_scale_zero_detaches_backbone(self):
        torch.manual_seed(0)
        cfg = StateProbeConfig(grad_scale=0.0)
        probe = StateProbe(input_dim=8, cfg=cfg)
        hidden = torch.randn(2, 3, 8, requires_grad=True)
        losses = probe.losses(hidden, distance_targets=torch.rand(2, 3),
                              return_targets=torch.rand(2, 3))
        sum(losses.values()).backward()
        assert hidden.grad is None or float(hidden.grad.abs().max()) == 0.0


def test_metrics_meters_scale_and_masking():
    import torch
    from longnav.utils.state_probe import StateProbe, StateProbeConfig
    cfg = StateProbeConfig.from_dict({
        "readout_offset": -2, "grad_scale": 0.1,
        "distance": {"hidden_dims": [8], "n_bins": 16, "d_max": 40.0,
                     "hl_sigma_ratio": 0.75, "loss_weight": 1.0},
        "value": {"hidden_dims": [8], "n_bins": 11, "v_min": -8.0, "v_max": 24.0,
                  "hl_sigma_ratio": 0.75, "loss_weight": 1.0, "gamma": 0.97}})
    probe = StateProbe(4, cfg)
    h = torch.randn(1, 5, 4)
    d = torch.tensor([[1.0, 2.0, float("nan"), 4.0, 5.0]])
    r = torch.tensor([[0.5, float("nan"), float("nan"), 1.5, 2.0]])
    m = probe.metrics(h, d, r)
    # counts respect each head's own mask
    assert float(m["probe_dist_n"]) == 4.0
    assert float(m["probe_value_n"]) == 3.0
    # mae consistent with its own sums, and finite
    assert torch.isfinite(m["probe_dist_abs_err_sum"])
    assert abs(float(m["probe_dist_err_sum"])) <= float(m["probe_dist_abs_err_sum"]) + 1e-6
    # predictions live in meter space: an untrained head's expectation is
    # inside [0, d_max], so MAE against 1-5 m targets is bounded by d_max
    assert float(m["probe_dist_abs_err_sum"]) / 4.0 < 40.0
    # all-NaN targets -> no keys rather than zeros that would skew a drain
    m2 = probe.metrics(h, torch.full((1, 5), float("nan")), None)
    assert "probe_dist_n" not in m2
