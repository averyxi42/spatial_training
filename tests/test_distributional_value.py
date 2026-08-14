"""DistributionalValueHead: target math, loss behavior, dispatch seams, GAE adapter.

Pure logic -- no model, no GPU required (runs on CPU tensors).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
import torch

from longnav.utils.vlm_worker import DistributionalValueHead, ValueHead


def make_head(**kw):
    kw.setdefault("input_dim", 8)
    kw.setdefault("hidden_dims", [16])
    kw.setdefault("n_bins", 51)
    kw.setdefault("v_min", -5.0)
    kw.setdefault("v_max", 15.0)
    return DistributionalValueHead(**kw)


class TestTargets:
    def test_targets_are_distributions(self):
        h = make_head()
        g = torch.tensor([[-4.0, 0.0, 7.3, 14.9]])
        t = h.targets(g)
        assert t.shape == (1, 4, 51)
        assert torch.allclose(t.sum(-1), torch.ones(1, 4), atol=1e-6)
        assert (t >= 0).all()

    def test_mean_preserved_in_range(self):
        # HL-Gauss's point: the target distribution's expectation recovers the return
        # (away from the support edges, where clamping bites by design).
        h = make_head()
        g = torch.linspace(-3.0, 13.0, 25)
        means = h.targets(g) @ h.bin_centers
        assert torch.allclose(means, g, atol=0.02), (means - g).abs().max()

    def test_out_of_range_clamps_to_edges(self):
        h = make_head()
        t = h.targets(torch.tensor([-100.0, 100.0]))
        assert h.bin_centers[t[0].argmax()] == h.bin_centers[0]
        assert h.bin_centers[t[1].argmax()] == h.bin_centers[-1]

    def test_value_inverts_targets(self):
        # Feeding target log-probs back as logits must recover the return: value() and
        # targets() are inverse up to the smearing, which mean-preserves in range.
        h = make_head()
        g = torch.tensor([2.5, 9.0])
        logits = torch.log(h.targets(g).clamp_min(1e-12))
        assert torch.allclose(h.value(logits), g, atol=0.02)


class TestLoss:
    def test_loss_minimized_at_correct_prediction(self):
        h = make_head()
        g = torch.tensor([[3.0]])
        mask = torch.ones(1, 1)
        right = torch.log(h.targets(g).clamp_min(1e-12))
        wrong = torch.log(h.targets(torch.tensor([[9.0]])).clamp_min(1e-12))
        assert h.distributional_loss(right, g, mask) < h.distributional_loss(wrong, g, mask)

    def test_masked_tokens_do_not_contribute(self):
        h = make_head()
        g = torch.tensor([[3.0, 500.0]])          # second token nonsense but masked out
        logits = torch.randn(1, 2, 51)
        mask = torch.tensor([[1.0, 0.0]])
        only_first = h.distributional_loss(logits[:, :1], g[:, :1], mask[:, :1])
        assert torch.allclose(h.distributional_loss(logits, g, mask), only_first, atol=1e-6)

    def test_gradient_flows_and_is_finite(self):
        h = make_head()
        x = torch.randn(4, 8)
        logits = h(x)
        loss = h.distributional_loss(logits.unsqueeze(0), torch.full((1, 4), 5.0),
                                     torch.ones(1, 4))
        loss.backward()
        grads = [p.grad for p in h.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)

    def test_ce_descends_to_target(self):
        # A few SGD steps on a free logit vector must move value() toward the return --
        # the end-to-end property regression critics get wrong under scale shift.
        h = make_head()
        logits = torch.zeros(1, 1, 51, requires_grad=True)
        g = torch.tensor([[11.0]])
        opt = torch.optim.SGD([logits], lr=5.0)
        for _ in range(200):
            opt.zero_grad()
            h.distributional_loss(logits, g, torch.ones(1, 1)).backward()
            opt.step()
        assert abs(float(h.value(logits)[0, 0]) - 11.0) < 0.2


class TestSeams:
    def test_capability_flags(self):
        assert make_head().is_distributional is True
        # ValueHead's default dtype is the string 'float32' (production converts before
        # constructing); pass a real dtype here.
        assert not getattr(ValueHead(8, [16], dtype=torch.float32),
                           "is_distributional", False)

    def test_forward_emits_bins_not_scalar(self):
        h = make_head()
        out = h(torch.randn(2, 3, 8))
        assert out.shape == (2, 3, 51)
        v = h.value(out)
        assert v.shape == (2, 3)
        assert (v >= -5.0).all() and (v <= 15.0).all()   # expectation stays on support

    def test_config_refusals(self):
        with pytest.raises(ValueError):
            make_head(v_min=5.0, v_max=5.0)
        with pytest.raises(ValueError):
            make_head(n_bins=1)


class TestGaeAdapter:
    def test_matches_verl_gae(self):
        from longnav.utils.rl_core import compute_gae_config_advantage
        from verl.trainer.ppo.core_algos import compute_gae_advantage_return

        class Cfg:
            gamma = 0.95
            def get(self, k, d=None):
                return {"lam": 0.95}.get(k, d)

        torch.manual_seed(0)
        r = torch.randn(3, 7)
        v = torch.randn(3, 7)
        m = torch.ones(3, 7); m[1, 5:] = 0; m[2, 3:] = 0
        adv, ret = compute_gae_config_advantage(
            token_level_rewards=r, response_mask=m, values=v, config=Cfg())
        adv2, ret2 = compute_gae_advantage_return(
            token_level_rewards=r, values=v * m, response_mask=m, gamma=0.95, lam=0.95)
        assert torch.allclose(adv, adv2) and torch.allclose(ret, ret2)

    def test_refuses_missing_values(self):
        from longnav.utils.rl_core import compute_gae_config_advantage

        class Cfg:
            gamma = 0.95
            def get(self, k, d=None): return d

        with pytest.raises(ValueError):
            compute_gae_config_advantage(
                token_level_rewards=torch.zeros(1, 3),
                response_mask=torch.ones(1, 3), values=None, config=Cfg())
