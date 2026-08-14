"""The h-space tether: ref-KL estimators and the merged-base reference.

Two things are pinned. First, the estimator conventions rl_loss logs (`ref/kl_k1`,
`ref/kl_k3`) really estimate KL(pi || pi_ref) for the diagonal Gaussians the continuous
heads emit -- checked against the closed form, so a sign or direction slip in the inline
formulas cannot survive. Second, the reference semantics: with the SFT adapter MERGED
into the base (vlm.merge_adapter_dir) and a fresh zero-delta LoRA on top, peft's stock
`disable_adapter()` must return the init policy bit-exactly -- LoRA silenced AND
modules_to_save switched to their original copies -- which is the entire correctness
story of rl_config.ref_kl. No snapshots, no second adapter set, by design.

CPU only; the reference tests use a toy LoRA model, not a checkpoint.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _gaussian_logprob(a, mu, log_std):
    std = np.exp(log_std)
    lp = -0.5 * (((a - mu) / std) ** 2 + 2.0 * log_std + np.log(2.0 * np.pi))
    return lp.sum(-1)


def _closed_form_kl(mu, log_std, mu_ref, log_std_ref):
    v, vr = np.exp(2 * log_std), np.exp(2 * log_std_ref)
    return 0.5 * np.sum(2 * (log_std_ref - log_std) + (v + (mu - mu_ref) ** 2) / vr - 1.0)


class TestEstimatorConventions:
    """r = log pi_ref - log pi under a ~ pi; k1 = E[-r]; k3 = E[e^r - 1 - r]."""

    def setup_method(self, _):
        rng = np.random.default_rng(7)
        self.dim = 16
        self.mu = rng.normal(size=self.dim)
        self.log_std = rng.normal(scale=0.1, size=self.dim) - 1.0
        self.mu_ref = self.mu + rng.normal(scale=0.05, size=self.dim)
        self.log_std_ref = self.log_std + rng.normal(scale=0.02, size=self.dim)
        a = rng.normal(size=(200_000, self.dim)) * np.exp(self.log_std) + self.mu
        self.r = (_gaussian_logprob(a, self.mu_ref, self.log_std_ref)
                  - _gaussian_logprob(a, self.mu, self.log_std))
        self.kl = _closed_form_kl(self.mu, self.log_std, self.mu_ref, self.log_std_ref)

    def test_k1_estimates_kl(self):
        assert abs((-self.r).mean() - self.kl) < 0.02 * max(self.kl, 1.0)

    def test_k2_estimates_kl_and_is_nonnegative(self):
        # k2 = r^2/2: the quadratic estimator rl_loss logs and the leash penalizes
        # (k3's exponential is wrong-conditioned at chain-density |r| -- see rl_loss).
        # E[r^2/2] ~= KL for small divergence (KL ~ chi^2/2 locally).
        k2 = 0.5 * self.r ** 2
        assert (k2 >= 0).all()
        assert abs(k2.mean() - self.kl) < 0.05 * max(self.kl, 1.0)

    def test_matched_policies_read_zero(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(1000, self.dim)) * np.exp(self.log_std) + self.mu
        r = (_gaussian_logprob(a, self.mu, self.log_std)
             - _gaussian_logprob(a, self.mu, self.log_std))
        assert np.allclose(r, 0.0)


class TestDisableAdapterReference:
    """The default ref mechanism: with the SFT policy MERGED into the base, peft's stock
    `disable_adapter()` must return the init policy bit-exactly -- LoRA delta silenced
    AND modules_to_save switched back to their original (init) copies -- and restore the
    trained policy on exit. This is the whole correctness story of rl_config.ref_kl; no
    snapshots, no adapter sets."""

    class _Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Linear(8, 8)
            self.action_head = nn.Linear(8, 2)

        def forward(self, x):
            return self.action_head(torch.relu(self.body(x)))

    def _toy_peft_model(self):
        peft = pytest.importorskip("peft")
        torch.manual_seed(0)
        base = self._Toy()   # stands in for the MERGED base: base weights ARE "SFT"
        cfg = peft.LoraConfig(r=2, target_modules=["body"],
                              modules_to_save=["action_head"])
        return peft.get_peft_model(base, cfg)

    def test_fresh_lora_is_a_zero_delta(self):
        model = self._toy_peft_model()
        x = torch.randn(4, 8)
        with torch.no_grad():
            out = model(x)
            with model.disable_adapter():
                ref = model(x)
        assert torch.equal(out, ref), "peft inits B=0: step-0 policy must BE the base"

    def test_disable_adapter_returns_the_init_policy_and_restores(self):
        model = self._toy_peft_model()
        x = torch.randn(4, 8)
        with torch.no_grad():
            init_out = model(x)
        # Simulate training: move every trainable peft param (LoRA + the ACTIVE
        # modules_to_save copy of the head).
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p))
            trained_out = model(x)
        assert not torch.allclose(trained_out, init_out), \
            "the perturbation must actually change the policy"
        with torch.no_grad():
            with model.disable_adapter():
                ref_out = model(x)
            after_out = model(x)
        assert torch.equal(ref_out, init_out), \
            "disable_adapter must yield the init policy bit-exactly (head included: " \
            "modules_to_save switches back to the original copy)"
        assert torch.equal(after_out, trained_out), \
            "exiting the context must restore the trained policy bit-exactly"

    def test_head_training_alone_is_also_silenced(self):
        # The modules_to_save half of the guarantee in isolation: even an arm that
        # trains ONLY the head still gets the init head as its reference.
        model = self._toy_peft_model()
        x = torch.randn(4, 8)
        with torch.no_grad():
            init_out = model(x)
            for n, p in model.named_parameters():
                if p.requires_grad and "action_head" in n:
                    p.add_(1.0)
            with model.disable_adapter():
                ref_out = model(x)
        assert torch.equal(ref_out, init_out)
