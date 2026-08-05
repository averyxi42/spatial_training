"""`PlanarSE2Encoder` -- the properties it exists to guarantee.

Every test here corresponds to a measured defect in `FourierSE2Encoder`, which is
kept and unchanged. The point of this file is that those defects stay fixed, so
each test names the number it is defending.
"""

import math

import numpy as np
import pytest
import torch

from longnav.utils.modality_embed import (
    FourierSE2Encoder,
    ModalityEmbedSpec,
    PlanarSE2Encoder,
    build_encoder,
)


def poses(n=512, seed=0):
    """Relative poses on roughly the corpus's scale: xy std ~3 m, heading uniform."""
    g = torch.Generator().manual_seed(seed)
    xy = torch.randn(n, 2, generator=g) * 3.0
    th = (torch.rand(n, 1, generator=g) * 2 - 1) * math.pi
    return torch.cat([xy, th], dim=1)


# ---------------------------------------------------------------------------
# the output norm is set, not hoped for
# ---------------------------------------------------------------------------


def test_output_norm_is_exactly_out_norm():
    """The old encoder's LayerNorm was documented as bounding this and could not:
    it sits upstream of an unbounded projection. v5 reached a median norm of 16.5
    against 0.362 for the tokens actually in a sequence."""
    for target in (0.4, 1.0, 2.5):
        enc = PlanarSE2Encoder(3, 128, out_norm=target, gain_init=1.0)
        norms = enc(poses()).norm(dim=1)
        assert torch.allclose(norms, torch.full_like(norms, target), atol=1e-5)


def test_output_norm_survives_a_wildly_rescaled_trunk():
    """Norm control must not depend on the trunk staying well behaved -- that is
    the whole failure mode it replaces."""
    enc = PlanarSE2Encoder(3, 128, out_norm=1.0, gain_init=1.0)
    with torch.no_grad():
        for module in enc.net:
            if isinstance(module, torch.nn.Linear):
                module.weight *= 1000.0
    norms = enc(poses()).norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


# ---------------------------------------------------------------------------
# gradients: the v6 death must not be reproducible here
# ---------------------------------------------------------------------------


def test_gradient_is_alive_at_init_unlike_fourier_plus_max_norm():
    """The regression test for the bug that froze `run_v6_regress_pose` for its
    whole life: zero output -> zero gradient -> weights never move."""
    dead = FourierSE2Encoder(3, 64, max_norm=1.0, zero_init=True)
    dead(poses(64)).sum().backward()
    assert all(float(p.grad.abs().sum()) == 0.0 for p in dead.parameters()), \
        "the known-dead configuration should still be dead; this test is calibration"

    enc = PlanarSE2Encoder(3, 64, gain_init=0.0)
    out = enc(poses(64))
    assert float(out.norm()) == 0.0, "gain_init=0 must still emit exactly zero"
    out.sum().backward()
    assert float(enc.out_gain.grad.abs().sum()) > 0.0, \
        "the gain must learn on the first backward even though the output is zero"


def test_trunk_is_gated_for_one_step_not_forever():
    """`gain_init=0` gates the trunk while the gain is zero. It must leave zero
    immediately -- v6's encoder never did."""
    torch.manual_seed(0)
    enc = PlanarSE2Encoder(3, 32, gain_init=0.0)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3)
    target = torch.randn(64, 32)
    trunk_grads = []
    for _ in range(3):
        loss = ((enc(poses(64)) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        trunk_grads.append(float(enc.net[0].weight.grad.abs().sum()))
        opt.step()
    assert trunk_grads[0] == 0.0
    assert trunk_grads[1] > 0.0 and trunk_grads[2] > 0.0


def test_a_positive_gain_is_live_from_step_zero():
    enc = PlanarSE2Encoder(3, 32, gain_init=0.1)
    out = enc(poses(64))
    assert float(out.norm(dim=1).mean()) == pytest.approx(0.1, abs=1e-5)
    out.sum().backward()
    assert float(enc.net[0].weight.grad.abs().sum()) > 0.0


def test_gradient_is_finite_at_the_origin():
    """(0, 0, 0) is row 0 of every episode, so it is the single most common input."""
    enc = PlanarSE2Encoder(3, 32, gain_init=1.0)
    enc(torch.zeros(4, 3)).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in enc.parameters())


# ---------------------------------------------------------------------------
# the balance defect: heading was 1% of the varying energy
# ---------------------------------------------------------------------------


def _sensitivity(enc, P):
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        base = enc(P)
        turned = P.clone()
        turned[:, 2] = (torch.rand(len(P), generator=g) * 2 - 1) * math.pi
        moved = P.clone()
        moved[:, :2] = P[torch.randperm(len(P), generator=g), :2]
        return (float((enc(turned) - base).norm(dim=1).mean()),
                float((enc(moved) - base).norm(dim=1).mean()))


def test_heading_and_position_carry_comparable_energy():
    """Measured on the trained FourierSE2Encoder: randomising heading moved the
    output 1.16, randomising position 10.54 -- heading was ~1% of the energy, and
    a 180 deg flip moved it 3.6x LESS than a 10 cm slide."""
    P = poses(1024)
    heading, position = _sensitivity(PlanarSE2Encoder(3, 256, gain_init=1.0), P)
    share = heading ** 2 / (heading ** 2 + position ** 2)
    assert 0.2 < share < 0.8, f"heading share of varying energy is {share:.3f}"


def test_reversing_heading_moves_more_than_a_small_step():
    P = poses(512)
    enc = PlanarSE2Encoder(3, 256, gain_init=1.0)
    with torch.no_grad():
        base = enc(P)
        flipped = P.clone()
        flipped[:, 2] = torch.remainder(flipped[:, 2] + 2 * math.pi, 2 * math.pi) - math.pi
        nudged = P.clone()
        nudged[:, 0] += 0.10
        flip_move = float((enc(flipped) - base).norm(dim=1).mean())
        nudge_move = float((enc(nudged) - base).norm(dim=1).mean())
    assert flip_move > nudge_move


def test_embedding_distance_tracks_pose_distance():
    """FourierSE2Encoder measured -0.218 on real poses: further apart in metres,
    CLOSER in embedding space. Aliasing, from a 0.25 m basis under a 0.40 m step."""
    P = poses(1024)
    enc = PlanarSE2Encoder(3, 256, gain_init=1.0)
    with torch.no_grad():
        out = enc(P).numpy()
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(P), 4000), rng.integers(0, len(P), 4000)
    d_pose = np.linalg.norm(P[i, :2].numpy() - P[j, :2].numpy(), axis=1)
    d_emb = np.linalg.norm(out[i] - out[j], axis=1)
    assert np.corrcoef(d_pose, d_emb)[0, 1] > 0.3


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_features_are_x_y_cos_sin_by_default():
    enc = PlanarSE2Encoder(3, 32)
    assert enc.feature_dim == 4
    f = enc.features(torch.tensor([[4.0, 0.0, 0.0]]))
    assert f[0].tolist() == pytest.approx([1.0, 0.0, 1.0, 0.0], abs=1e-6)


def test_harmonics_and_radius_widen_the_basis():
    assert PlanarSE2Encoder(3, 32, n_heading_harmonics=3).feature_dim == 8
    assert PlanarSE2Encoder(3, 32, use_radius=True).feature_dim == 5


def test_scales_are_buffers_so_they_travel_with_the_checkpoint():
    """Training and rollout must not be able to disagree about what a metre is."""
    enc = PlanarSE2Encoder(3, 32, pos_scale=4.0, out_norm=0.4)
    state = enc.state_dict()
    assert float(state["pos_scale"]) == pytest.approx(4.0)
    assert float(state["out_norm"]) == pytest.approx(0.4)
    other = PlanarSE2Encoder(3, 32, pos_scale=1.0, out_norm=1.0)
    other.load_state_dict(state)
    assert float(other.pos_scale) == pytest.approx(4.0)
    assert float(other.out_norm) == pytest.approx(0.4)


def test_rejects_incoherent_configuration():
    with pytest.raises(ValueError, match="n_features must be 3"):
        PlanarSE2Encoder(4, 32)
    with pytest.raises(ValueError, match="pos_scale"):
        PlanarSE2Encoder(3, 32, pos_scale=0.0)
    with pytest.raises(ValueError, match="out_norm"):
        PlanarSE2Encoder(3, 32, out_norm=-1.0)
    with pytest.raises(ValueError, match="n_heading_harmonics"):
        PlanarSE2Encoder(3, 32, n_heading_harmonics=0)


def test_builds_from_the_shipped_spec():
    import json
    import os

    path = "/Projects/spatial_training/dump/pose_injection/pose_spec_planar.json"
    if not os.path.isfile(path):
        pytest.skip(f"{path} not present")
    spec = ModalityEmbedSpec.from_dict(json.load(open(path))[0])
    enc = build_encoder(spec, 2048)
    assert isinstance(enc, PlanarSE2Encoder)
    assert enc.feature_dim == 4
    assert float(enc.pos_scale) == 4.0
