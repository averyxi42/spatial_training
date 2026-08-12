"""The latent split must be the identity at init, or nothing measured after it means anything.

`docs/LATENT_RL.md` calls the parity assertion the one cheap plumbing check this design gets:
with `W_mu = I`, `b_mu = 0`, a zero-init posterior and `latent_mode="mean"`, a warm-started
model is NUMERICALLY IDENTICAL to the deterministic one it started from. That single identity
also covers where the split sits in the forward pass and whether the checkpoint loaded.

CPU only, no data, no checkpoints -- the identity is a property of the initialisation and the
wiring, not of any trained weights.
"""

import sys
from pathlib import Path

import pytest
import torch

# `longnav` is a namespace package (no `src/longnav/__init__.py`), so a bare import resolves
# to whichever checkout is first on the path. Pin this one before importing.
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from longnav.utils.latent_intent import (  # noqa: E402
    LatentConfig, LatentSplit, PosteriorEncoder, kl_shared_sigma, latent_diagnostics,
)

DIM, N, T = 64, 5, 20


def _h(seed=0):
    return torch.randn(N, DIM, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------------------
# The parity identity
# ---------------------------------------------------------------------------------------
class TestParityAtInit:
    def test_mean_mode_is_bit_identical_to_h(self):
        """Not `allclose` -- exactly equal. A tolerance here would hide a real reordering."""
        split, h = LatentSplit(DIM), _h()
        c = split.draw(h, mode="mean")["c"]
        assert torch.equal(c, h), (c - h).abs().max().item()

    def test_sigma_is_sigma0_everywhere_at_init(self):
        split = LatentSplit(DIM, sigma0=0.03)
        sigma = split.draw(_h(), mode="mean")["sigma"]
        assert torch.allclose(sigma, torch.full_like(sigma, 0.03), atol=1e-7)

    def test_posterior_shift_is_exactly_zero_at_init(self):
        """`KL = 0` at step 0 is what stops a fresh posterior dragging `mu` off the warm
        start to chase noise. Zero, not small."""
        post = PosteriorEncoder(dim=DIM, n_ticks=T)
        d = post(_h(), torch.randn(N, T, 3))
        assert torch.equal(d, torch.zeros_like(d))
        assert float(kl_shared_sigma(d, torch.zeros(N, DIM)).abs().max()) == 0.0

    def test_posterior_shift_does_not_perturb_parity(self):
        """The two inits compose: split identity + zero posterior => still exactly `h`."""
        split, post, h = LatentSplit(DIM), PosteriorEncoder(dim=DIM, n_ticks=T), _h()
        d = post(h.detach(), torch.randn(N, T, 3))
        assert torch.equal(split.draw(h, mode="mean", delta_mu=d)["c"], h)

    def test_sample_mode_departs_from_h(self):
        """Guards the opposite failure: a `sample` mode that silently returns the mean would
        pass every parity test above and give RL nothing to explore with."""
        split, h = LatentSplit(DIM, sigma0=0.1), _h()
        c = split.draw(h, mode="sample")["c"]
        assert not torch.equal(c, h)
        assert 0.05 < float((c - h).std()) < 0.2


# ---------------------------------------------------------------------------------------
# The rotation, which must preserve the identity exactly
# ---------------------------------------------------------------------------------------
class TestRotation:
    @staticmethod
    def _orthogonal(seed=0):
        a = torch.randn(DIM, DIM, generator=torch.Generator().manual_seed(seed))
        return torch.linalg.qr(a)[0]

    def test_rotation_preserves_the_parity_identity(self):
        """`R^T R h = h` is the whole reason a PCA basis is free to adopt."""
        split, h = LatentSplit(DIM, rotation=self._orthogonal()), _h()
        c = split.draw(h, mode="mean")["c"]
        assert torch.allclose(c, h, atol=1e-5), (c - h).abs().max().item()

    def test_non_orthogonal_rotation_is_refused(self):
        with pytest.raises(ValueError, match="orthogonal"):
            LatentSplit(DIM, rotation=torch.randn(DIM, DIM))

    def test_rotation_makes_noise_full_covariance_in_h_space(self):
        """Diagonal in the rotated basis is NOT diagonal in `h`'s -- that is the point."""
        R = self._orthogonal()
        split = LatentSplit(DIM, sigma0=0.1, rotation=R)
        with torch.no_grad():                       # make sigma genuinely anisotropic
            split.to_log_sigma.bias.copy_(torch.linspace(-4.0, 0.0, DIM))
        h = _h(1)[:1].expand(4000, DIM)         # one observation, many draws
        noise = split.draw(h, mode="sample")["c"] - split.draw(h, mode="mean")["c"]
        cov = (noise.T @ noise) / noise.shape[0]
        off = cov - torch.diag(torch.diag(cov))
        assert float(off.abs().max()) > 1e-3, "rotated sigma should correlate h-space dims"


# ---------------------------------------------------------------------------------------
# The KL and its diagnostics
# ---------------------------------------------------------------------------------------
class TestSharedSigmaKL:
    def test_matches_the_general_diagonal_gaussian_kl_when_scales_are_shared(self):
        d = torch.randn(N, DIM)
        log_sigma = torch.randn(N, DIM) * 0.3 - 2.0
        s2 = (2 * log_sigma).exp()
        general = (log_sigma - log_sigma + (s2 + d.pow(2)) / (2 * s2) - 0.5).sum(dim=-1)
        assert torch.allclose(kl_shared_sigma(d, log_sigma), general, atol=1e-5)

    def test_kl_is_zero_only_when_the_shift_is_zero(self):
        log_sigma = torch.full((N, DIM), -3.0)
        assert float(kl_shared_sigma(torch.zeros(N, DIM), log_sigma).max()) == 0.0
        assert float(kl_shared_sigma(torch.full((N, DIM), 1e-3), log_sigma).min()) > 0.0

    def test_wider_sigma_makes_a_given_shift_cheaper(self):
        """The gradient direction that stops `sigma` collapsing: KL falls as sigma grows,
        opposing the reconstruction term's pull downward."""
        d = torch.full((N, DIM), 0.1)
        narrow = kl_shared_sigma(d, torch.full((N, DIM), -3.0)).mean()
        wide = kl_shared_sigma(d, torch.full((N, DIM), -1.0)).mean()
        assert float(wide) < float(narrow)

    def test_active_dims_counts_only_dims_carrying_information(self):
        split = LatentSplit(DIM, sigma0=0.1)
        pieces = split.draw(_h(), mode="sample")
        d = torch.zeros(N, DIM)
        d[:, :7] = 1.0                                   # 7 dims well above 0.01 nats
        m = latent_diagnostics(pieces, d, kl_shared_sigma(d, pieces["log_sigma"]), _h())
        assert int(m["sum_active_dims"] / N) == 7

    def test_no_decay_parameters_are_the_output_layer(self):
        """Zero-init plus weight decay pulls `delta_mu` back to zero, which IS the
        degenerate solution. The optimiser must be able to name these."""
        post = PosteriorEncoder(dim=DIM, n_ticks=T)
        named = {id(p) for p in post.no_decay_parameters()}
        assert named == {id(post.out.weight), id(post.out.bias)}


# ---------------------------------------------------------------------------------------
# Off is off
# ---------------------------------------------------------------------------------------
class TestDisabled:
    def test_config_default_is_off(self):
        assert LatentConfig().enabled is False
        assert "OFF" in LatentConfig().describe()

    def test_bad_mode_raises_rather_than_silently_meaning_mean(self):
        with pytest.raises(ValueError, match="mode must be"):
            LatentSplit(DIM).draw(_h(), mode="argmax")

    def test_posterior_refuses_a_mis_shaped_chunk(self):
        with pytest.raises(ValueError, match="actions must be"):
            PosteriorEncoder(dim=DIM, n_ticks=T)(_h(), torch.randn(N, T + 1, 3))
