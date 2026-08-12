"""`reset(anchor_offset=...)` must move the pose frame and change NOTHING else.

The diagnostic it exists for: `relative_se2` anchors every injected pose on row 0 of the
accumulated column, so "does the same task get harder when the anchor is further away" can
only be asked by changing what row 0 is. The temptation is to insert an extra pose turn to
move it -- which would also change the context the model reads (an extra `<pose>` marker,
extra text, a shifted position id for everything after it) and confound the very thing the
experiment measures.

`seed_anchor` instead pushes a fictitious row into the accumulator that is never emitted,
because `accumulate_pose_rows` returns `relative_se2_tail(..., len(rows))` -- the tail only.
These tests pin exactly that:

  * the number of emitted rows per call is identical with and without the offset, so the
    marker/row binding is untouched and no extra marker is needed;
  * the emitted headings are bit-identical, so the manipulation is translation-only and
    `relative_se2`'s rotation into the anchor frame is unchanged;
  * the emitted positions differ by exactly the SE(2) transform the offset implies;
  * off is off -- `anchor_offset=None` leaves the accumulator empty and every value equal
    to what it was before this existed.

Run against a stub carrying nothing but the accumulator state, which is what
`accumulate_pose_rows`'s docstring says the production code supports, so none of this needs
a multi-billion-parameter backbone.
"""

import numpy as np
import pytest
import torch

from longnav.utils.pose_frame import relative_se2
from longnav.utils.vector_rollout import VectorRolloutPolicy


class Accumulator:
    """The pose-accumulator half of `VectorRolloutPolicy`, and nothing else."""

    def __init__(self, anchor_offset=None):
        VectorRolloutPolicy.reset(self, anchor_offset=anchor_offset)

    # The two production methods, bound to this stub.
    pose_values = VectorRolloutPolicy.pose_values
    pose_rows = VectorRolloutPolicy.pose_rows


# A PointNav-shaped call sequence: an announcement writing (agent, goal), then plain
# per-observation agent rows, then another announcement.
ROUTE = [
    [(1.0, 2.0, 0.30), (5.0, 2.5, 0.0)],
    [(1.4, 2.1, 0.35)],
    [(2.0, 2.2, 0.40)],
    [(4.8, 2.5, 0.10), (9.0, -1.0, 0.0)],
    [(5.2, 2.4, 0.05)],
]


def run(offset):
    acc = Accumulator(anchor_offset=offset)
    return [acc.pose_rows(rows).clone() for rows in ROUTE], acc


def test_off_leaves_the_accumulator_exactly_as_it_was():
    out, acc = run(None)
    assert len(acc._raw_poses) == sum(len(r) for r in ROUTE), "no phantom row when off"
    flat = [p for rows in ROUTE for p in rows]
    expected = relative_se2(torch.tensor(np.array(flat, dtype=np.float64)))
    got = torch.cat(out, dim=0)
    assert torch.allclose(got, expected, atol=1e-6)
    assert torch.allclose(got[0], torch.zeros(3), atol=1e-7), "row 0 is the origin when off"


def test_the_row_count_per_call_is_identical_with_and_without_an_offset():
    """The property that keeps the context unchanged: same rows out, same markers needed."""
    plain, _ = run(None)
    moved, _ = run((12.0, -7.0))
    assert [tuple(t.shape) for t in plain] == [tuple(t.shape) for t in moved]
    assert sum(t.shape[0] for t in moved) == sum(len(r) for r in ROUTE)


def test_the_seeded_row_is_never_emitted():
    _plain, acc = run((12.0, -7.0))
    assert len(acc._raw_poses) == sum(len(r) for r in ROUTE) + 1, "the anchor row is stored"
    moved, _ = run((12.0, -7.0))
    assert sum(t.shape[0] for t in moved) == sum(len(r) for r in ROUTE), "but not returned"


def test_headings_are_bit_identical_so_the_manipulation_is_translation_only():
    plain = torch.cat(run(None)[0], dim=0)
    moved = torch.cat(run((12.0, -7.0))[0], dim=0)
    assert torch.equal(plain[:, 2], moved[:, 2])


def test_positions_move_by_exactly_the_offset_rotated_into_the_anchor_frame():
    """The anchor keeps the first pose's heading, so the rotation is the same and the
    positions shift by a single constant vector -- the offset expressed in that frame."""
    plain = torch.cat(run(None)[0], dim=0)
    moved = torch.cat(run((12.0, -7.0))[0], dim=0)
    delta = moved[:, :2] - plain[:, :2]
    assert torch.allclose(delta, delta[0].expand_as(delta), atol=1e-5), (
        "every pose must shift by the SAME vector; a varying shift would mean the "
        "rotation changed too and the experiment would not isolate distance"
    )
    # Moving the anchor to a + d sends R(-theta0)(p - a) to R(-theta0)(p - a) - R(-theta0)d,
    # so the shift is the NEGATED offset expressed in the anchor's frame.
    assert delta[0].norm() == pytest.approx(np.hypot(12.0, -7.0), abs=1e-3)
    theta0 = ROUTE[0][0][2]
    dx, dy = 12.0, -7.0
    c, s = np.cos(-theta0), np.sin(-theta0)
    want = -torch.tensor([c * dx - s * dy, s * dx + c * dy], dtype=torch.float32)
    assert torch.allclose(delta[0], want, atol=1e-4)


def test_a_heading_offset_rotates_the_cloud_and_shifts_every_heading():
    """The second axis: orientation relative to the anchor, with position held fixed."""
    dth = np.pi / 3
    plain = torch.cat(run(None)[0], dim=0)
    turned = torch.cat(run((0.0, 0.0, dth))[0], dim=0)
    # Headings all shift by -dtheta (wrapped).
    diff = torch.remainder(plain[:, 2] - turned[:, 2] + np.pi, 2 * np.pi) - np.pi
    assert torch.allclose(diff, torch.full_like(diff, float(dth)), atol=1e-5)
    # Positions rotate by -dtheta about the origin, so their norms are preserved.
    assert torch.allclose(plain[:, :2].norm(dim=1), turned[:, :2].norm(dim=1), atol=1e-4)
    c, s = np.cos(-dth), np.sin(-dth)
    rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    assert torch.allclose(turned[:, :2], plain[:, :2] @ rot.T, atol=1e-4)


def test_a_heading_offset_leaves_the_row_count_alone_too():
    plain, _ = run(None)
    turned, _ = run((0.0, 0.0, np.pi))
    assert [tuple(t.shape) for t in plain] == [tuple(t.shape) for t in turned]


def test_a_two_element_offset_still_works_and_means_no_rotation():
    two = torch.cat(run((3.0, 4.0))[0], dim=0)
    three = torch.cat(run((3.0, 4.0, 0.0))[0], dim=0)
    assert torch.equal(two, three)


def test_a_malformed_offset_is_refused():
    with pytest.raises(ValueError, match="anchor_offset"):
        Accumulator(anchor_offset=(1.0,))
    with pytest.raises(ValueError, match="anchor_offset"):
        Accumulator(anchor_offset=(1.0, 2.0, 3.0, 4.0))


@pytest.mark.parametrize("dist", [0.0, 5.0, 20.0, 50.0])
def test_every_pose_shifts_by_the_offset_exactly(dist):
    """What the experiment manipulates: a rigid translation of the whole pose cloud.

    Asserted exactly rather than through the mean norm, which cannot grow by the full
    offset -- the triangle inequality caps it, and at 50 m against poses spread over ~10 m
    the mean grows by only ~44. The rigid shift is the real invariant.
    """
    plain = torch.cat(run(None)[0], dim=0)
    moved = torch.cat(run((dist, 0.0))[0], dim=0)
    theta0 = ROUTE[0][0][2]
    want = -torch.tensor([np.cos(-theta0) * dist, np.sin(-theta0) * dist],
                         dtype=torch.float32)
    assert torch.allclose(moved[:, :2] - plain[:, :2], want.expand_as(plain[:, :2]),
                          atol=1e-4)


def test_a_large_offset_does_push_the_whole_cloud_far_from_the_origin():
    """Not asserted per-offset: a shift comparable to the cloud's own extent can move part
    of it *closer* to the origin. Once the offset dominates that extent it cannot."""
    plain = torch.cat(run(None)[0], dim=0)
    extent = float(plain[:, :2].norm(dim=1).max())
    for dist in (3 * extent, 10 * extent):
        moved = torch.cat(run((dist, 0.0))[0], dim=0)
        assert float(moved[:, :2].norm(dim=1).min()) > dist - extent - 1e-3
