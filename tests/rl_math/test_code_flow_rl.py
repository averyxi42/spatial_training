"""CodeFlowHead (Path A0): sampler/scorer parity, the table actuator, and the plumbing.

CPU-only, tiny stack: a real `FlowActionDecoder` + `FlowActionCodec` with a real
`CodePolicyHead`/`CodeContextMixer` attached, so every tensor path is the shipped one.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from longnav.utils.code_conditioned_head import CodeContextMixer, CodePolicyHead
from longnav.utils.code_flow_rl import CodeFlowHead
from longnav.utils.flow_matching_head import FlowActionCodec, FlowActionDecoder
from longnav.utils.rl_core import collate_trajectories

N_XY, N_TH, D_MODEL, N_TOK, CTX = 4, 5, 8, 8, 64
V = N_XY * N_TH


def _head(T=0.5, seed=0):
    torch.manual_seed(0)
    dec = FlowActionDecoder(context_dim=CTX, n_ticks=6, d_model=D_MODEL,
                            n_context_tokens=N_TOK, n_layers=1, n_heads=2)
    codec = FlowActionCodec(dec, num_inference_steps=4)
    codec.code_mixer = CodeContextMixer(N_XY, N_TH, context_dim=CTX, d_model=D_MODEL,
                                        n_tokens=N_TOK, tok_xy=2, tok_theta=2)
    codec.code_head = CodePolicyHead(N_XY, N_TH, CTX, kind="mlp", hidden=32)
    readout = nn.Linear(16, CTX)
    h = CodeFlowHead(readout=readout, codec=codec, gap=3, policy_temperature=T)
    h.seed(seed)
    return h


def test_sampler_and_scorer_agree_on_the_stored_code():
    head = _head(T=0.5)
    hs = torch.randn(7, CTX)
    for i in range(7):
        chain, pos, lp, chunk = head.sample_chain_np(hs[i].numpy())
        scored = head.chain_log_prob_batch(hs[i].reshape(1, 1, -1),
                                           torch.tensor(chain).reshape(1, 1, 2),
                                           torch.tensor(pos).reshape(1, 1, 1))
        assert abs(float(scored) - lp) < 1e-5
        assert chunk.shape == (3, 3)
        assert pos.shape == (1,) and pos.dtype == np.int64


def test_scorer_is_differentiable_through_h_only():
    head = _head()
    hs = torch.randn(2, 3, CTX, requires_grad=True)
    chains = torch.tensor([[[1., 2.], [0., 4.], [3., 0.]],
                           [[2., 2.], [1., 1.], [0., 0.]]])
    lp = head.chain_log_prob_batch(hs, chains, torch.zeros(2, 3, 1))
    assert lp.shape == (2, 3)
    lp.sum().backward()
    assert hs.grad is not None and torch.isfinite(hs.grad).all()


def test_temperature_is_in_both_sampler_and_scorer():
    h_cold, h_warm = _head(T=0.25), _head(T=1.0)
    hs = torch.randn(1, 1, CTX)
    chain = torch.tensor([[[1., 2.]]])
    pos = torch.zeros(1, 1, 1)
    lp_c = h_cold.chain_log_prob_batch(hs, chain, pos)
    lp_w = h_warm.chain_log_prob_batch(hs, chain, pos)
    assert not torch.allclose(lp_c, lp_w)
    # entropy of the tempered policy is lower at lower T
    e = lambda hd: torch.distributions.Categorical(
        logits=hd._logits(hs.reshape(1, -1))).entropy()
    assert float(e(h_cold)) < float(e(h_warm))


def test_force_ode_is_deterministic_argmax():
    head = _head()
    head.force_ode = True
    h = np.random.RandomState(0).randn(CTX).astype(np.float32)
    c1, _, lp1, k1 = head.sample_chain_np(h)
    c2, _, lp2, k2 = head.sample_chain_np(h)
    assert (c1 == c2).all() and lp1 == lp2 and np.allclose(k1, k2)
    logits = head._logits(torch.tensor(h).reshape(1, -1))
    assert int(c1[0]) * N_TH + int(c1[1]) == int(logits.argmax())


def test_table_matches_a_direct_decode_and_decode_action_matches_sampler():
    head = _head()
    table = head._ensure_table()
    assert table.shape == (V, 6, 3)
    # direct decode of one code with r = 0 and z0 = 0 equals the table row
    mixer = head.codec.code_mixer
    cx, ct = torch.tensor([2]), torch.tensor([3])
    code = torch.cat([mixer.emb_xy(cx), mixer.emb_theta(ct)], dim=-1)
    ctx = torch.cat([code, code.new_zeros(1, mixer.n_reserved * mixer.d_model)], -1)
    from longnav.utils.bin_codec import compose_chunk
    head.codec.decoder.eval()          # the table builds under the eval pin; match it
    with torch.no_grad():
        direct = compose_chunk(head.codec.generate(ctx.float(),
                                                   noise=torch.zeros(1, 6, 3)))[0]
    assert torch.allclose(direct.float(), table[2 * N_TH + 3], atol=1e-5)
    # decode_action on the stored pair reproduces the sampled chunk
    h = np.random.RandomState(1).randn(CTX).astype(np.float32)
    chain, _, _, chunk = head.sample_chain_np(h)
    assert np.allclose(head.decode_action(chain), chunk, atol=1e-6)


def test_codes_survive_collate_as_float32_exactly():
    ep = lambda n, seed: {
        "actions_continuous": np.stack([np.array([i % N_XY, (i * 7) % N_TH], np.float32)
                                        for i in range(n)]),
        "sde_positions": np.zeros((n, 1), np.int64),
        "rollout_logprobs": np.full(n, -1.0, np.float32),
        "rewards": np.zeros(n, np.float32),
        "old_log_prob": np.full(n, -1.0, np.float32),
    }
    batch = collate_trajectories([ep(4, 0), ep(2, 1)], device="cpu")
    a = batch["actions_continuous"]
    assert a.shape == (2, 4, 2)
    assert torch.equal(a, a.round())                      # codes exact in float32
    assert batch["response_mask"].sum() == 6


def test_rtc_prefix_refused_loudly():
    head = _head()
    with pytest.raises(RuntimeError, match="A1"):
        head.sample_chain_np(np.zeros(CTX, np.float32), prefix=np.zeros((2, 3)))
    with pytest.raises(RuntimeError, match="A1"):
        head.chain_log_prob_batch(torch.zeros(1, 1, CTX), torch.zeros(1, 1, 2),
                                  torch.zeros(1, 1, 1),
                                  prefix_actions=torch.zeros(1, 1, 2, 3),
                                  prefix_len=torch.zeros(1, 1))


def test_sampling_parity_with_the_denormalize_seam():
    """The RL sampler draws the same codes the eval seam would, seeded identically."""
    head = _head(T=1.0, seed=123)
    codec = head.codec
    hs = torch.randn(5, CTX)
    # eval seam: codec.denormalize's sample branch (code_mode="sample", T=1)
    codec.code_mode = "sample"
    codec.code_temperature = 1.0
    gen = torch.Generator().manual_seed(123)
    codec.code_generator = gen
    seam_codes = []
    with torch.no_grad():
        for i in range(5):
            codec.denormalize(hs[i:i + 1])
            seam_codes.append(codec.last_code[0].tolist())
    head.seed(123)
    rl_codes = []
    for i in range(5):
        chain, *_ = head.sample_chain_np(hs[i].numpy())
        rl_codes.append([int(chain[0]), int(chain[1])])
    assert rl_codes == seam_codes


def test_overlay_modes_selected_first_then_topk_alternatives():
    head = _head(T=0.5)
    head.overlay_modes_k = 3
    h = np.random.RandomState(2).randn(CTX).astype(np.float32)
    chain, _, _, chunk = head.sample_chain_np(h)
    m = head.last_mode_chunks
    assert m is not None and m.shape == (4, 6, 3)          # selected + 3 alternatives
    table = head._ensure_table()
    sel = int(chain[0]) * N_TH + int(chain[1])
    # row 0 IS the executed code's chunk (its prefix is what the env executed)
    assert np.allclose(m[0], table[sel].cpu().numpy(), atol=1e-6)
    assert np.allclose(m[0][: head.gap], chunk, atol=1e-6)
    # rows 1.. are the top-K by pi_T with the selected code excluded
    logits = head._logits(torch.tensor(h).reshape(1, -1))
    alts = [j for j in torch.topk(logits[0], 4).indices.tolist() if j != sel][:3]
    assert np.allclose(m[1:], table[torch.tensor(alts)].cpu().numpy(), atol=1e-6)
    assert head.last_mode_probs.shape == (4,)
    # off by default: a fresh head fills nothing
    h2 = _head()
    h2.sample_chain_np(h)
    assert h2.last_mode_chunks is None
