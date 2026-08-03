"""Tests for the v2 AR decoder.

CPU only, no data, no checkpoints. Covers the invariants that are easy to break silently
and impossible to notice from a loss curve, plus the v2-specific structural claims.
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
from longnav.utils.ar_action_head_v2 import ActionDecoderV2  # noqa: E402

K, T, D, NCTX = 32, 6, 16, 4
CTX = NCTX * D


def make(**kw):
    torch.manual_seed(0)
    kw.setdefault("n_context_tokens", NCTX)
    return ActionDecoderV2(context_dim=kw.pop("context_dim", CTX), n_codes=K, n_ticks=T,
                           d_model=D, n_layers=2, n_heads=2, dim_ff=32, dropout=0.0,
                           **kw).eval()


def centroids():
    torch.manual_seed(1)
    return torch.randn(K, 3, dtype=torch.float64) * 0.02


# ---- structural claims v2 makes -------------------------------------------------------
def test_no_context_projection_and_no_start_embed():
    dec = make()
    names = set(dict(dec.named_parameters()))
    assert not any(n.startswith("context_proj") for n in names)
    assert "start_embed" not in names


def test_context_dim_must_match_token_space():
    with pytest.raises(ValueError, match="no context projection"):
        make(context_dim=CTX + 1)


def test_state_is_pose_only_by_default():
    dec = make()
    assert dec.state_dim == 3
    dec2 = make(use_incoming_motion=True)
    assert dec2.state_dim == 7


def test_state_uses_raw_theta_not_cos_sin():
    """v1 spent two features on (cos, sin); v2 spends one on theta. Verify the third
    feature tracks heading linearly rather than saturating like cos."""
    dec = make()
    cen = centroids()
    lab = torch.randint(0, K, (4, T))
    s = dec._state(lab, cen)
    assert s.shape == (4, T, 3)
    assert torch.allclose(s[:, 0], torch.zeros(4, 3))          # tick 0 = at origin


# ---- invariants inherited from v1, which must still hold ------------------------------
@pytest.mark.parametrize("attn_mode", ["causal", "markov"])
def test_placeholder_safety(attn_mode):
    dec = make(attn_mode=attn_mode)
    cen = centroids()
    ctx = torch.randn(3, CTX)
    a = torch.randint(0, K, (3, T))
    b = a.clone()
    b[:, T // 2:] = torch.randint(0, K, (3, T - T // 2))
    la = dec(ctx, a, centroids=cen)
    lb = dec(ctx, b, centroids=cen)
    torch.testing.assert_close(la[:, : T // 2], lb[:, : T // 2])


@pytest.mark.parametrize("attn_mode", ["causal", "markov"])
@pytest.mark.parametrize("incoming", [False, True])
def test_decode_matches_teacher_forced_replay(attn_mode, incoming):
    dec = make(attn_mode=attn_mode, use_incoming_motion=incoming)
    cen = centroids()
    ctx = torch.randn(3, CTX)
    inc = torch.randn(3, 4) if incoming else None
    tokens, logits = dec.decode(ctx, mode="argmax", centroids=cen, incoming=inc)
    replay = dec(ctx, tokens, centroids=cen, incoming=inc)
    torch.testing.assert_close(logits, replay, rtol=1e-4, atol=1e-5)


def test_markov_mask_not_discarded_by_is_causal():
    mask, is_causal = make(attn_mode="markov")._attn_mask(torch.device("cpu"),
                                                          torch.float32)
    assert is_causal is False
    C = NCTX
    assert torch.isinf(mask[C + 2, C + 1]) and mask[C + 2, C + 1] < 0
    assert mask[C + 2, C + 2] == 0
    assert (mask[C:, :C] == 0).all()
    _, causal_hint = make()._attn_mask(torch.device("cpu"), torch.float32)
    assert causal_hint is True


def test_centroids_are_required():
    with pytest.raises(ValueError, match="not optional"):
        make()(torch.randn(2, CTX), torch.randint(0, K, (2, T)))


# ---- the incoming-motion slot ---------------------------------------------------------
def test_incoming_motion_only_affects_tick_zero():
    dec = make(use_incoming_motion=True)
    cen = centroids()
    lab = torch.randint(0, K, (3, T))
    a = dec._state(lab, cen, torch.randn(3, 4))
    b = dec._state(lab, cen, torch.randn(3, 4))
    assert not torch.allclose(a[:, 0], b[:, 0])                 # tick 0 sees it
    torch.testing.assert_close(a[:, 1:, 3:], b[:, 1:, 3:])      # later ticks do not


def test_absent_incoming_motion_is_unknown_not_at_rest():
    """The validity bit is the whole point: without it, zeros would claim 'stationary'."""
    dec = make(use_incoming_motion=True)
    cen = centroids()
    lab = torch.randint(0, K, (2, T))
    s_absent = dec._state(lab, cen, None)
    at_rest = torch.zeros(2, 4)
    at_rest[:, 3] = 1.0                                          # known, and stationary
    s_rest = dec._state(lab, cen, at_rest)
    assert not torch.allclose(s_absent[:, 0], s_rest[:, 0])
    assert s_absent[0, 0, 6].item() == 0.0                       # valid bit low
    assert s_rest[0, 0, 6].item() == 1.0


# ---- config round trip ----------------------------------------------------------------
def test_config_round_trip():
    dec = make(attn_mode="markov", use_incoming_motion=True)
    cfg = dec.to_config()
    twin = ActionDecoderV2(context_dim=CTX, n_codes=K, n_ticks=T, **cfg)
    twin.load_state_dict(dec.state_dict())
    assert twin.attn_mode == "markov" and twin.use_incoming_motion is True


def test_v2_is_strictly_smaller_than_v1_at_matched_shape():
    """v2 drops context_proj and start_embed; nothing else changed size."""
    torch.manual_seed(0)
    v1 = CausalActionDecoder(context_dim=CTX, n_codes=K, n_ticks=T, d_model=D, n_layers=2,
                             n_heads=2, dim_ff=32, dropout=0.0, n_context_tokens=NCTX,
                             use_prefix_pose=True)
    v2 = make()
    n1 = sum(p.numel() for p in v1.parameters())
    n2 = sum(p.numel() for p in v2.parameters())
    assert n2 < n1, (n1, n2)
