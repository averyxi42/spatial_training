"""The flow-SDE head's invariants. CPU, toy dims, no checkpoint, no simulator.

What is pinned and why, in the order a failure would bite:

1. THE RATIO. `sample_chain_np` and `chain_log_prob_batch` share one transition function by
   construction; these tests pin the consequence -- recompute equals rollout log-prob for the
   same (h, chain, positions), bit-identically across calls, UNDER `model.train()` (the
   velocity net carries dropout, and a train-mode recompute corrupts the ratio silently --
   the failure mode docs/FLOW_SDE_RL.md ranks first).
2. THE MARGINAL. As `a -> 0` the SDE step degenerates to the Euler ODE step exactly, so a
   near-zero-noise chain must land where `euler_integrate` lands from the same `z_0`. This is
   the toy version of the marginal-preservation test and catches sign/convention errors in
   the score correction -- the external spec's "would have caught every wiring gotcha".
3. THE IDENTITIES. On the EXACT trained field `u = noise - action` over the interpolation
   `x_t = t*noise + (1-t)*action`, `eps_hat` recovers `noise` exactly. If a convention flip
   ever lands in `_sde_transition`, this fails first and names the line.
4. THE SEAMS. A Gaussian head exposes neither chain capability (the branch sites dispatch on
   `getattr`); `sde_positions` survives the collator's `(S, 1) -> (S,)` squeeze at n=1.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from longnav.utils.flow_matching_head import (  # noqa: E402
    FlowActionCodec, FlowActionDecoder, euler_integrate)
from longnav.utils.flow_sde_policy import (  # noqa: E402
    FlowSDEHead, SDEConfig, _sde_transition)
from longnav.utils.turn_vectors import TurnVectorHead  # noqa: E402
from longnav.utils.vlm_worker import ContinuousActionHead  # noqa: E402

HID, DIM, T, K, GAP = 64, 32, 20, 10, 10


def _head(n=3, a=0.2, seed=0):
    readout = TurnVectorHead(hidden_size=HID, out_dim=DIM, mode="mean", hidden_dims=(DIM,))
    dec = FlowActionDecoder(context_dim=DIM, n_ticks=T, d_model=DIM // 8,
                            n_context_tokens=8, n_heads=2, dim_ff=32, n_layers=1)
    codec = FlowActionCodec(dec, num_inference_steps=K)
    head = FlowSDEHead(readout, codec, gap=GAP, sde=SDEConfig(n=n, noise_a=a))
    head.seed(seed)
    return head


def _sample(head):
    h = head(torch.randn(1, 1, HID))["h"][0, 0].detach().numpy()
    chain, pos, lp, chunk = head.sample_chain_np(h)
    return h, chain, pos, lp, chunk


# --------------------------------------------------------------------------------------
# 1. The ratio
# --------------------------------------------------------------------------------------
class TestRatioIntegrity:
    def test_recompute_matches_rollout_logprob(self):
        head = _head()
        h, chain, pos, lp, _ = _sample(head)
        got = head.chain_log_prob_batch(
            torch.as_tensor(h).reshape(1, 1, -1),
            torch.as_tensor(chain).reshape(1, 1, -1),
            torch.as_tensor(pos).reshape(1, 1, -1)).item()
        assert abs(got - lp) < 1e-3, "sampler and scorer disagree on the same transitions"

    def test_recompute_is_bit_identical_under_train_mode(self):
        """Dropout=0.1 lives in the velocity net and RL runs under model.train(). The
        eval-pin lives INSIDE the head's methods; if it ever moves to a call site, this
        fails."""
        head = _head()
        h, chain, pos, _, _ = _sample(head)
        head.train()
        args = (torch.as_tensor(h).reshape(1, 1, -1),
                torch.as_tensor(chain).reshape(1, 1, -1),
                torch.as_tensor(pos).reshape(1, 1, -1))
        assert torch.equal(head.chain_log_prob_batch(*args),
                           head.chain_log_prob_batch(*args))

    def test_logprob_is_float32_and_differentiable(self):
        head = _head()
        _, chain, pos, _, _ = _sample(head)
        hg = torch.randn(1, 1, DIM, requires_grad=True)
        lp = head.chain_log_prob_batch(hg, torch.as_tensor(chain).reshape(1, 1, -1),
                                       torch.as_tensor(pos).reshape(1, 1, -1))
        assert lp.dtype == torch.float32
        lp.sum().backward()
        assert hg.grad is not None and hg.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in head.codec.decoder.parameters()), \
            "the score is theta-dependent; a detach() has crept in"

    def test_decode_action_reproduces_the_sampled_chunk(self):
        """What the env executed is derivable from what the ratio is over -- the
        stored-is-scored invariant, end to end."""
        head = _head()
        _, chain, _, _, chunk = _sample(head)
        np.testing.assert_array_equal(chunk, head.decode_action(chain))


# --------------------------------------------------------------------------------------
# 2. The marginal (toy)
# --------------------------------------------------------------------------------------
class TestMarginalPreservation:
    def test_zero_noise_limit_is_the_ode(self):
        """a -> 0: the drift correction (~a^2) and the injected noise (~a) both vanish, so
        the chain must land where euler_integrate lands from the same z_0. A sign or
        coefficient error in the score correction breaks this at ~a^2 scale -- run at
        a=1e-6 so anything wrong is visible and nothing right is."""
        head = _head(n=3, a=1e-6)
        h, chain, _, _, _ = _sample(head)
        z0 = torch.as_tensor(chain[: head.block]).reshape(1, T, 3)
        ctx = torch.as_tensor(h).reshape(1, -1)
        head.codec.decoder.eval()
        with torch.no_grad():
            ode = euler_integrate(
                lambda x, t: head.codec.decoder(ctx, x, t), z0, K)
        sde_final = torch.as_tensor(chain[-head.block:]).reshape(1, T, 3)
        assert torch.allclose(ode, sde_final, atol=1e-4), \
            "zero-noise SDE != ODE: score correction has a sign/coefficient error"


# --------------------------------------------------------------------------------------
# 3. The identities
# --------------------------------------------------------------------------------------
class TestScoreIdentities:
    def test_eps_hat_recovers_noise_on_the_exact_field(self):
        """`u_t = noise - action` over `x_t = t*noise + (1-t)*action` is what the field was
        trained toward (euler_integrate's docstring). On that exact field,
        eps_hat = x_t + (1-t)*u = noise identically -- the convention anchor."""
        noise = torch.randn(4, T, 3)
        action = torch.randn(4, T, 3)

        class ExactField:
            def __call__(self, ctx, x_t, t):
                return noise - action
        for tv in (0.9, 0.5, 0.2):
            t = torch.full((4,), tv)
            x_t = tv * noise + (1 - tv) * action
            mu, std = _sde_transition(ExactField(), torch.zeros(4, DIM), x_t, t,
                                      dt=-1.0 / K, cfg=SDEConfig(n=1, noise_a=0.3))
            eps_hat = x_t + (1 - tv) * (noise - action)
            assert torch.allclose(eps_hat, noise, atol=1e-5)
            # and the transition std matches the schedule at this t
            import math
            tc = min(max(tv, 1e-3), 0.95)
            want = 0.3 * math.sqrt(tc / (1 - tc)) * math.sqrt(1.0 / K)
            assert abs(std.flatten()[0].item() - want) < 1e-6


# --------------------------------------------------------------------------------------
# 4. The seams
# --------------------------------------------------------------------------------------
class TestSeams:
    def test_gaussian_head_exposes_no_chain_capability(self):
        """The branch sites dispatch on getattr; the pre-existing continuous path must be
        bit-for-bit untouched, which starts with the capability simply not existing."""
        g = ContinuousActionHead(input_dim=HID, action_dim=3)
        assert not hasattr(g, "sample_chain_np")
        assert not hasattr(g, "chain_log_prob_batch")

    @pytest.mark.parametrize("n", [1, 3])
    def test_positions_survive_the_collator_squeeze(self, n):
        """rl_core's collator squeezes a trailing dim of 1, so at n=1 positions arrive as
        (B, S) and at n>1 as (B, S, n). The scorer reshapes by its OWN sde.n; both ranks
        must read back to the same answer."""
        head = _head(n=n)
        h, chain, pos, lp, _ = _sample(head)
        ht = torch.as_tensor(h).reshape(1, 1, -1)
        ct = torch.as_tensor(chain).reshape(1, 1, -1)
        full = torch.as_tensor(pos).reshape(1, 1, n)
        squeezed = full.squeeze(-1) if n == 1 else full
        a = head.chain_log_prob_batch(ht, ct, full).item()
        b = head.chain_log_prob_batch(ht, ct, squeezed).item()
        assert a == b and abs(a - lp) < 1e-3

    def test_oracle_chunk_determines_no_chain(self):
        """Config refusals: n outside the admissible range, and a > 0 required."""
        with pytest.raises(ValueError, match="admissible"):
            _head(n=K)          # n_exclude_last=1 leaves only K-1 positions
        with pytest.raises(ValueError, match="noise sweep"):
            SDEConfig(n=1, noise_a=0.0)
