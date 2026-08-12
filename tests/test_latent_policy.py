"""The latent head is a policy over `c`; the existing continuous path must not notice it.

Two things are being pinned. First, that `LatentIntentHead` satisfies
`ContinuousActionHead`'s contract exactly -- same keys, same shapes -- so `rollout_core`'s
continuous branch needs no special-casing. Second, and more important, that adding it changed
NOTHING for `gaussian_head`: the dispatch resolves to the same class, the classmethod
constructs it with the same arguments, the actuator hook is identity because no existing head
defines `decode_action`, and the log-prob reduction still defaults to a plain sum.

CPU only. The head is exercised against hand-built modules rather than a checkpoint, because
what is under test is the wiring and the contract, not any trained weights.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# `longnav` is a namespace package; pin this checkout before importing. See tests/conftest.py.
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from longnav.utils.flow_matching_head import FlowActionCodec, FlowActionDecoder  # noqa: E402
from longnav.utils.latent_intent import LatentSplit  # noqa: E402
from longnav.utils.latent_policy import LatentIntentHead  # noqa: E402
from longnav.utils.rollout_core import reduce_gaussian_logprob  # noqa: E402
from longnav.utils.turn_vectors import TurnVectorHead  # noqa: E402
from longnav.utils.vlm_worker import ContinuousActionHead  # noqa: E402

HIDDEN, DIM, TICKS, GAP = 64, 32, 20, 10


def _head(gap=GAP, sigma0=0.02, pin=0):
    readout = TurnVectorHead(hidden_size=HIDDEN, out_dim=DIM, mode="mean", hidden_dims=(DIM,))
    decoder = FlowActionDecoder(context_dim=DIM, n_ticks=TICKS, d_model=DIM // 8,
                                n_context_tokens=8, n_heads=2, dim_ff=32, n_layers=1)
    codec = FlowActionCodec(decoder, num_inference_steps=3,
                            latent=LatentSplit(dim=DIM, sigma0=sigma0))
    codec.pin_flow_noise(pin)
    return LatentIntentHead(readout=readout, codec=codec, gap=gap)


# ---------------------------------------------------------------------------------------
# The contract rollout_core relies on
# ---------------------------------------------------------------------------------------
class TestPolicyContract:
    def test_returns_the_same_keys_and_shapes_as_the_gaussian_head(self):
        hidden = torch.randn(1, 1, HIDDEN)
        ours = _head()(hidden)
        theirs = ContinuousActionHead(input_dim=HIDDEN, action_dim=DIM)(hidden)
        assert set(ours) == set(theirs) == {"mu", "log_std"}
        for k in ("mu", "log_std"):
            assert ours[k].shape == theirs[k].shape == (1, 1, DIM)

    def test_sigma_comes_from_the_split_not_a_constant(self):
        """A head that re-initialises log_std has discarded the CVAE. sigma0=0.02 must show
        up as log_std ~ log(0.02), not as the Gaussian head's -0.5 default."""
        out = _head(sigma0=0.02)(torch.randn(1, 1, HIDDEN))
        assert torch.allclose(out["log_std"], torch.full_like(out["log_std"], np.log(0.02)),
                              atol=1e-4)

    def test_log_std_floor_is_far_below_the_gaussian_default(self):
        """The ELBO fits sigma near 1e-3 (log_std ~ -6.9). A -5.0 floor would clamp it and
        silently inflate exploration."""
        assert _head().min_log_std <= -20.0


# ---------------------------------------------------------------------------------------
# The actuator
# ---------------------------------------------------------------------------------------
class TestActuator:
    def test_decode_returns_exactly_gap_rows(self):
        chunk = _head(gap=GAP).decode_action(np.random.randn(DIM).astype(np.float32))
        assert chunk.shape == (GAP, 3), "the chunk tail must be discarded, as in training"

    def test_decode_is_deterministic_given_c(self):
        """Pinned base noise is what makes `c -> chunk` a fixed map. Without it `old_log_prob`
        would describe an action that never executed."""
        head, c = _head(), np.random.randn(DIM).astype(np.float32)
        assert np.array_equal(head.decode_action(c), head.decode_action(c))

    def test_different_c_gives_a_different_chunk(self):
        """Guards the opposite failure: a decode that ignores `c` would pass the determinism
        test above and hand RL an action space with no effect.

        The two `c` are RANDOM, not `zeros` vs `ones`. Constant vectors differ only in each
        128-block's mean, and the decoder is `norm_first`, so that difference is stripped
        before the action tokens read it -- they decode identically, and a test built on them
        fails for a reason that has nothing to do with the code under test. This is the
        attenuated-directions effect `docs/LATENT_RL.md` warns about, observed.
        """
        head = _head()
        rng = np.random.default_rng(0)
        a = head.decode_action(rng.normal(size=DIM).astype(np.float32))
        b = head.decode_action(rng.normal(size=DIM).astype(np.float32))
        assert not np.allclose(a, b)

    def test_constant_c_directions_are_attenuated_by_the_decoder_prenorm(self):
        """The flip side, pinned deliberately: perturbing `c` along a per-block constant is
        very nearly a no-op. This is why the spread gate is measured on BEHAVIOUR and never
        on ||dc||, and why ~16 of 1024 dims will read as inactive for structural reasons."""
        head = _head()
        base = head.decode_action(np.zeros(DIM, dtype=np.float32))
        shifted = head.decode_action(np.full(DIM, 3.0, dtype=np.float32))
        assert np.allclose(base, shifted, atol=1e-5)

    def test_unpinned_noise_is_refused_at_construction(self):
        cfg = {"checkpoint_dir": "x", "gap": GAP, "action_space_dim": DIM}
        with pytest.raises(ValueError, match="pin_flow_noise_seed"):
            LatentIntentHead.from_policy_head_config(cfg, input_dim=HIDDEN,
                                                     dtype=torch.float32)

    def test_a_deterministic_checkpoint_is_refused(self):
        decoder = FlowActionDecoder(context_dim=DIM, n_ticks=TICKS, d_model=DIM // 8,
                                    n_context_tokens=8, n_heads=2, dim_ff=32, n_layers=1)
        with pytest.raises(ValueError, match="no latent split"):
            LatentIntentHead(readout=TurnVectorHead(hidden_size=HIDDEN, out_dim=DIM),
                             codec=FlowActionCodec(decoder), gap=GAP)


# ---------------------------------------------------------------------------------------
# The existing paths, which must not have moved
# ---------------------------------------------------------------------------------------
class TestGaussianPathUnchanged:
    def test_classmethod_builds_what_the_inline_call_built(self):
        cfg = {"action_space_dim": 2, "gaussian_init_log_std": -0.5,
               "gaussian_min_log_std": -5.0, "gaussian_max_log_std": 2.0}
        torch.manual_seed(1234)
        inline = ContinuousActionHead(input_dim=HIDDEN, action_dim=2, init_log_std=-0.5,
                                      min_log_std=-5.0, max_log_std=2.0,
                                      dtype=torch.float32)
        torch.manual_seed(1234)
        viacls = ContinuousActionHead.from_policy_head_config(cfg, input_dim=HIDDEN,
                                                             dtype=torch.float32)
        for a, b in zip(inline.state_dict().values(), viacls.state_dict().values()):
            assert torch.equal(a, b), "same seed, same construction => identical weights"

    def test_gaussian_head_has_no_decode_action_so_the_hook_is_identity(self):
        """`rollout_core` calls `decode_action` only when the head defines it. This is the
        assertion that keeps the seam a no-op for every pre-existing path."""
        assert not hasattr(ContinuousActionHead(input_dim=HIDDEN, action_dim=2),
                           "decode_action")

    def test_reduction_defaults_to_a_plain_sum(self):
        lp = np.random.randn(1, 7).astype(np.float32)
        assert np.allclose(reduce_gaussian_logprob(lp), np.sum(lp, axis=-1))
        assert np.allclose(reduce_gaussian_logprob(lp, "sum"), np.sum(lp, axis=-1))

    def test_mean_reduction_must_be_asked_for_and_is_not_the_sum(self):
        lp = np.random.randn(1, 7).astype(np.float32)
        assert not np.allclose(reduce_gaussian_logprob(lp, "mean"), np.sum(lp, axis=-1))

    def test_unknown_reduction_raises_rather_than_falling_back(self):
        with pytest.raises(ValueError, match="logprob_reduction"):
            reduce_gaussian_logprob(np.zeros((1, 3)), "logsumexp")
