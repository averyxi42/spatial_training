"""Tests for the multi-token context, prefix-pose and markov-mask decoder variants.

CPU only, no data, no checkpoints -- pure module-level assertions on the pieces that are
easy to get subtly wrong and impossible to notice from a loss curve:

  * decode-time placeholder safety for BOTH channels (tokens and pose features)
  * the markov mask actually blocking cross-tick information
  * `is_causal` not silently discarding a custom mask
  * backward compatibility of old 1-D `start_embed` checkpoints
  * rank-preserving context split

Run: <any env with torch>  python -m pytest tests/test_ar_head_variants.py
"""

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from longnav.utils.ar_action_head import CausalActionDecoder  # noqa: E402

K, T, D, CTX = 32, 6, 16, 64


def make(**kw):
    torch.manual_seed(0)
    dec = CausalActionDecoder(context_dim=kw.pop("context_dim", CTX), n_codes=K, n_ticks=T,
                              d_model=D, n_layers=2, n_heads=2, dim_ff=32, dropout=0.0, **kw)
    return dec.eval()


def centroids():
    torch.manual_seed(1)
    return torch.randn(K, 3, dtype=torch.float64) * 0.02


@pytest.mark.parametrize("pose", [False, True])
@pytest.mark.parametrize("n_ctx", [1, 4])
def test_placeholder_safety(pose, n_ctx):
    """Logits at tick t must not depend on labels at positions > t.

    This is the invariant `decode` relies on to be correct without a KV cache. With
    prefix-pose on it now covers a second channel, since the pose features are composed
    from the same label tensor.
    """
    dec = make(n_context_tokens=n_ctx, use_prefix_pose=pose,
               context_dim=n_ctx * D if n_ctx > 1 else CTX)
    cen = centroids()
    ctx = torch.randn(3, n_ctx * D if n_ctx > 1 else CTX)
    a = torch.randint(0, K, (3, T))
    b = a.clone()
    b[:, T // 2:] = torch.randint(0, K, (3, T - T // 2))  # perturb the FUTURE only

    la = dec(ctx, a, centroids=cen)
    lb = dec(ctx, b, centroids=cen)
    # positions strictly before the perturbation must be bit-identical
    torch.testing.assert_close(la[:, : T // 2], lb[:, : T // 2])


def test_markov_mask_blocks_the_attention_path():
    """With the state channel OFF, markov mode must sever cross-tick information entirely.

    Tick t's input is then only `token_embed(label_{t-1}) + tick_embed(t)`, so changing
    label 0 may move tick 1 (which consumes it) and nothing after.
    """
    dec = make(use_prefix_pose=False, attn_mode="markov")
    cen = centroids()
    ctx = torch.randn(3, CTX)
    a = torch.randint(0, K, (3, T))
    b = a.clone()
    b[:, 0] = (b[:, 0] + 1) % K
    la, lb = dec(ctx, a, centroids=cen), dec(ctx, b, centroids=cen)
    torch.testing.assert_close(la[:, 2:], lb[:, 2:])


def test_markov_with_pose_propagates_only_through_the_state():
    """With prefix-pose ON, markov mode deliberately DOES let tick 0 reach later ticks --
    the accumulated pose contains d_0. That is the point: history reaches the decision as
    a sufficient STATE rather than as an attendable sequence. This test pins that
    behaviour so a future change that silently drops the pose channel is caught."""
    dec = make(use_prefix_pose=True, attn_mode="markov")
    cen = centroids()
    ctx = torch.randn(3, CTX)
    a = torch.randint(0, K, (3, T))
    b = a.clone()
    b[:, 0] = (b[:, 0] + 1) % K
    la, lb = dec(ctx, a, centroids=cen), dec(ctx, b, centroids=cen)
    assert not torch.allclose(la[:, 2:], lb[:, 2:]), \
        "pose channel should carry tick 0's effect into later ticks under markov mode"


def test_causal_mask_does_not_block_prefix():
    """Sanity counterpart: under 'causal' the same perturbation DOES propagate."""
    dec = make(use_prefix_pose=True, attn_mode="causal")
    cen = centroids()
    ctx = torch.randn(3, CTX)
    a = torch.randint(0, K, (3, T))
    b = a.clone()
    b[:, 0] = (b[:, 0] + 1) % K
    la = dec(ctx, a, centroids=cen)
    lb = dec(ctx, b, centroids=cen)
    assert not torch.allclose(la[:, 2:], lb[:, 2:]), \
        "causal mode should let tick 0 influence later ticks"


def test_markov_mask_is_not_discarded_by_is_causal():
    """`is_causal=True` permits the fast path to ignore a custom mask; the markov mode
    must therefore report False, or the mask silently does nothing."""
    dec = make(use_prefix_pose=True, attn_mode="markov")
    mask, is_causal = dec._attn_mask(torch.device("cpu"), torch.float32)
    assert is_causal is False
    C = dec.n_context_tokens
    assert torch.isinf(mask[C + 2, C + 1]) and mask[C + 2, C + 1] < 0   # tick 2 !-> tick 1
    assert mask[C + 2, C + 2] == 0                                       # tick 2 -> itself
    assert (mask[C:, :C] == 0).all()                                     # ticks -> context

    cmask, c_is_causal = make()._attn_mask(torch.device("cpu"), torch.float32)
    assert c_is_causal is True


def test_prefix_state_is_causal_and_zero_at_first_tick():
    dec = make(use_prefix_pose=True)
    cen = centroids()
    labels = torch.randint(0, K, (2, T))
    feat = dec._prefix_state(labels, cen)
    assert feat.shape == (2, T, dec.POSE_FEAT_DIM)
    # tick 0: no pose, no previous differential -> [0,0,cos0,sin0,0,0,0]
    expect = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(feat[:, 0], expect.expand(2, -1))
    # perturbing the last label cannot change any earlier tick's features
    b = labels.clone()
    b[:, -1] = (b[:, -1] + 1) % K
    torch.testing.assert_close(feat[:, :-1], dec._prefix_state(b, cen)[:, :-1])


def test_decode_matches_teacher_forced_on_its_own_output():
    """Free-running decode must produce logits identical to a teacher-forced pass over the
    tokens it actually emitted -- the strongest end-to-end check that the sequential loop
    feeds both channels back consistently."""
    for kw in ({}, {"use_prefix_pose": True},
               {"use_prefix_pose": True, "attn_mode": "markov"},
               {"n_context_tokens": 4, "context_dim": 4 * D, "use_prefix_pose": True},
               {"n_context_tokens": 4, "context_dim": 4 * D, "use_prefix_pose": True,
                "direct_context": True}):
        dec = make(**kw)
        cen = centroids()
        ctx = torch.randn(3, dec.context_dim)
        tokens, logits = dec.decode(ctx, mode="argmax", centroids=cen)
        replay = dec(ctx, tokens, centroids=cen)
        torch.testing.assert_close(logits, replay, rtol=1e-4, atol=1e-5)


def test_context_split_is_rank_preserving():
    dec = make(n_context_tokens=4, context_dim=4 * D)
    assert dec.context_proj.out_features == 4 * D
    ctx = torch.randn(5, 4 * D)
    toks = dec.context_proj(ctx).view(5, 4, D)
    assert toks.shape == (5, 4, D)
    assert dec.start_embed.shape == (4, D)


def test_direct_context_has_no_projection_and_reshapes():
    dec = make(n_context_tokens=4, context_dim=4 * D, direct_context=True)
    assert dec.context_proj is None
    names = dict(dec.named_parameters())
    assert not any(n.startswith("context_proj") for n in names)
    # equivalence: a projecting decoder whose context_proj is the identity must agree
    ref = make(n_context_tokens=4, context_dim=4 * D, direct_context=False)
    ref.load_state_dict({k: v for k, v in dec.state_dict().items()}, strict=False)
    with torch.no_grad():
        ref.context_proj.weight.copy_(torch.eye(4 * D))
        ref.context_proj.bias.zero_()
    cen = centroids()
    ctx = torch.randn(3, 4 * D)
    lab = torch.randint(0, K, (3, T))
    torch.testing.assert_close(dec(ctx, lab, centroids=cen), ref(ctx, lab, centroids=cen))


def test_direct_context_rejects_mismatched_width():
    # NB: CTX == 4*D here, so the mismatch has to be constructed explicitly.
    assert CTX == 4 * D
    with pytest.raises(ValueError, match="direct_context requires"):
        make(n_context_tokens=4, context_dim=4 * D + 1, direct_context=True)
    with pytest.raises(ValueError, match="direct_context requires"):
        make(n_context_tokens=3, context_dim=4 * D, direct_context=True)


def test_direct_context_round_trips_through_config():
    dec = make(n_context_tokens=4, context_dim=4 * D, direct_context=True,
               use_prefix_pose=True)
    cfg = dec.to_config()
    assert cfg["direct_context"] is True
    twin = CausalActionDecoder(context_dim=4 * D, n_codes=K, n_ticks=T, **cfg)
    twin.load_state_dict(dec.state_dict())
    assert twin.context_proj is None


def test_old_checkpoint_start_embed_upgrades():
    """Checkpoints written before multi-token context stored start_embed as (d_model,)."""
    old = make()
    sd = old.state_dict()
    sd["start_embed"] = sd["start_embed"].squeeze(0)      # simulate the old 1-D layout
    assert sd["start_embed"].dim() == 1
    fresh = make()
    fresh.load_state_dict(sd)                              # must not raise
    assert fresh.start_embed.shape == (1, D)


def test_markov_requires_pose_is_enforced_at_cli_level():
    """The decoder itself allows the combination; the guard lives in the train script, so
    assert the decoder at least builds and that the mask is what we think it is."""
    dec = make(attn_mode="markov")
    mask, is_causal = dec._attn_mask(torch.device("cpu"), torch.float32)
    assert is_causal is False and mask.shape == (dec.n_context_tokens + T,) * 2
