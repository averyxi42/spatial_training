"""Soft code targets: the distance matrix and the kernel, on a tiny tokenizer.

CPU-only, no checkpoint: a randomly initialised `DualTokenizer` is enough, because every
property checked here is about the construction (symmetry, units, the one-hot limit), not
about a trained codebook.
"""
import math

import pytest
import torch
import torch.nn as nn

from longnav.utils.chunk_tokenizer import DualTokenizer, FrozenChunkTokenizer
from longnav.utils.code_conditioned_head import (
    code_centroid_chunks, code_distance_matrix, soft_code_targets,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _tiny_tokenizer(seed=0, xy_levels=(3, 2), theta_levels=(2, 2), n_ticks=6):
    torch.manual_seed(seed)
    tok = FrozenChunkTokenizer.__new__(FrozenChunkTokenizer)
    nn.Module.__init__(tok)
    tok.model = DualTokenizer(list(xy_levels), list(theta_levels), n_ticks=n_ticks,
                              d_model=16, n_layers=1, n_head=2).eval()
    for p in tok.model.parameters():
        p.requires_grad_(False)
    tok.register_buffer("xy_scale", torch.tensor(0.35))
    tok.register_buffer("theta_scale", torch.tensor(0.38))
    tok.strict_wrap_guard = False
    tok.checkpoint_path = "<test>"
    return tok


SCALES = (0.03, 0.03, 0.05)


def test_centroids_follow_joint_index_layout():
    tok = _tiny_tokenizer()
    ch = code_centroid_chunks(tok)
    vx, vt = tok.model.vocab_xy, tok.model.vocab_theta
    assert ch.shape == (vx * vt, 6, 3)
    # row c_xy * n_theta + c_theta: the xy part depends on c_xy only, theta on c_theta only
    assert torch.allclose(ch[0 * vt + 1, :, :2], ch[0 * vt + 0, :, :2])
    assert torch.allclose(ch[1 * vt + 0, :, 2], ch[0 * vt + 0, :, 2])


def test_distance_matrix_is_a_metric_in_action_scale_units():
    tok = _tiny_tokenizer()
    d = code_distance_matrix(tok, SCALES)
    V = tok.model.vocab_xy * tok.model.vocab_theta
    assert d.shape == (V, V)
    assert torch.allclose(d, d.t())
    assert torch.all(d.diagonal() == 0)
    assert torch.all(d[~torch.eye(V, dtype=bool)] > 0)
    # scaling every action scale by k scales every distance by 1/k: the matrix IS in
    # action-scale units, which is what makes sigma comparable to the flow loss
    d2 = code_distance_matrix(tok, tuple(2 * s for s in SCALES))
    assert torch.allclose(d2, d / 2, atol=1e-5)


def test_distance_uses_equal_tick_weights():
    # RMS over all ticks and dims: the mean over ticks of the per-tick squared error.
    tok = _tiny_tokenizer()
    from longnav.utils.bin_codec import decompose_chunk
    ch = code_centroid_chunks(tok).double()
    x = decompose_chunk(ch) / torch.tensor(SCALES, dtype=torch.float64)
    ref = ((x[0] - x[1]) ** 2).mean().sqrt()
    assert torch.isclose(code_distance_matrix(tok, SCALES)[0, 1].double(), ref, atol=1e-5)


def test_grid_metric_is_lattice_l1():
    tok = _tiny_tokenizer()
    d = code_distance_matrix(tok, SCALES, metric="grid")
    vt = tok.model.vocab_theta
    # theta codes 0 and 1 differ by one lattice step in dim 0 at levels [2, 2]
    assert d[0, 1] == 1.0
    # xy code 0 vs 1 (one step) with the same theta
    assert d[0, vt] == 1.0


def test_soft_targets_recover_one_hot_as_sigma_vanishes():
    tok = _tiny_tokenizer()
    d = code_distance_matrix(tok, SCALES)
    tgt = torch.tensor([0, 5, 7])
    q = soft_code_targets(d, tgt, sigma=1e-3)
    assert torch.allclose(q, torch.nn.functional.one_hot(tgt, d.shape[0]).float(), atol=1e-6)
    q = soft_code_targets(d, tgt, sigma=0.5)
    assert torch.allclose(q.sum(-1), torch.ones(3))
    assert torch.all(q.argmax(-1) == tgt)                 # the true code stays the mode
    ent = -(q * q.clamp_min(1e-12).log()).sum(-1)
    assert torch.all(ent > 0) and torch.all(ent < math.log(d.shape[0]))


def test_soft_ce_matches_hard_ce_at_zero_width_and_is_smaller_near_misses():
    tok = _tiny_tokenizer()
    d = code_distance_matrix(tok, SCALES)
    V = d.shape[0]
    tgt = torch.tensor([3])
    # a prediction concentrated on the target's nearest neighbour
    nn_idx = (d[3] + torch.eye(V)[3] * 1e9).argmin()
    logits = torch.full((1, V), -10.0)
    logits[0, nn_idx] = 10.0
    hard = torch.nn.functional.cross_entropy(logits, tgt)
    soft = -(soft_code_targets(d, tgt, 1e-3) * torch.log_softmax(logits, -1)).sum()
    assert torch.isclose(hard, soft, atol=1e-4)
    softer = -(soft_code_targets(d, tgt, 1.0) * torch.log_softmax(logits, -1)).sum()
    assert softer < hard                                   # a near miss is cheaper


def test_metric_diagnostics_are_consistent():
    """The metric diagnostics on a synthetic (logits, target) batch, computed the same way
    `_code_metrics` does, agree with hand-computed values."""
    tok = _tiny_tokenizer()
    d = code_distance_matrix(tok, SCALES)
    V = d.shape[0]
    tgt = torch.tensor([2, 5])
    logits = torch.zeros(2, V)
    logits[0, 2] = 8.0                 # confident and right
    logits[1, 0] = 8.0                 # confident and wrong
    p = torch.softmax(logits, -1)
    pred = logits.argmax(-1)
    d_row = d.index_select(0, tgt)
    mdist_argmax = d_row.gather(1, pred[:, None]).squeeze(1)
    mdist_expected = (p * d_row).sum(-1)
    assert mdist_argmax[0] == 0.0 and mdist_argmax[1] == d[5, 0]
    assert mdist_expected[0] < mdist_expected[1]
    pmass = p.gather(1, tgt[:, None]).squeeze(1)
    assert pmass[0] > 0.9 and pmass[1] < 0.1
    near = (p * (d_row <= 0.2).float()).sum(-1)
    assert near[0] >= pmass[0]         # the true code is inside its own radius
