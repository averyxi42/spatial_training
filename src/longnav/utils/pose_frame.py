"""
SE(2) frame arithmetic for injected pose, and the one function both paths call.

The dataset stores raw **scene-frame** `obs_poses` -- `(x, y, theta)` with a per-scene
arbitrary origin. What the model is shown is the pose **relative to the first observation
in the visible context**, expressed in that observation's own frame. The conversion happens
at runtime rather than at dataset build time, which is what lets one rule cover both cases:

  * no windowing -- the first visible observation is the episode start;
  * windowing    -- the first visible observation is the window start.

A window *is* a new episode from the agent's perspective. It sees frames `k..k+n` and
nothing before them, so an origin outside that range is a number it cannot possibly
resolve. Anchoring to the first visible row means the same value always denotes the same
thing, in training and in deployment, whether or not the window moved. This is the decision
`dump/modality_embed/DESIGN.md` 8.1(1) left open.

Everything routes through `relative_se2`. The collator calls it on a whole window; the
rollout calls it on the raw poses accumulated so far and keeps the last row. Both get the
identical function on identical inputs, which is the point -- a train/rollout divergence in
the frame convention is silent (the model reads confident nonsense) and fatal to the
experiment. `tests/test_pose_injection.py` asserts the two agree row for row.

Heading is wrapped everywhere a difference is taken. On this corpus a naive
`theta - theta0` reaches 5.955 rad where the true rotation is 0.400 rad -- the same
quantity, off by `2*pi`, and nothing downstream would flag it.
"""

from __future__ import annotations

import math
from typing import Any

import torch

# `(x, y, theta)`. The width of one raw pose row and of one transformed row alike: the
# transform is a change of frame, not a change of representation. `(cos, sin)` expansion and
# Fourier features belong to the encoder, so an encoding can be ablated without rebuilding
# the dataset.
POSE_DIM = 3

TWO_PI = 2.0 * math.pi


def wrap_to_pi(theta: torch.Tensor) -> torch.Tensor:
    """Wrap an angle to `(-pi, pi]`.

    `pi - ((pi - theta) mod 2pi)` rather than the more common
    `((theta + pi) mod 2pi) - pi`: the latter is the half-open interval on the *other* side,
    `[-pi, pi)`, so it sends an exact half turn to `-pi`. Both are defensible; they differ
    only at the boundary, and picking one and stating it is what stops the boundary from
    being decided twice, differently, in two places.
    """
    return math.pi - torch.remainder(math.pi - theta, TWO_PI)


def relative_se2(poses: Any) -> torch.Tensor:
    """Raw scene-frame poses -> poses relative to row 0, in row 0's frame.

    `poses`: `(N, 3)` array-like of `(x, y, theta)`, in occurrence order.
    Returns `(N, 3)` float32. Row 0 is exactly `(0, 0, 0)`.

    Both parts of "relative" are done, and the rotation is the part that is easy to skip:

      * **translate** so the first observation is the origin. Without this the model reads
        the scene's arbitrary world origin, which is a usable fingerprint for scene
        identity -- a probe, or the model, can learn *which* scene it is rather than where
        it is in it.
      * **rotate** into the first observation's heading, so `+x` is "ahead at the start"
        rather than an arbitrary compass direction. Translation alone leaves the scene's
        global orientation in the value, which is the same leak in a different coordinate.

    Computed in float64 and returned as float32: the subtraction is between two absolute
    scene coordinates that may be far from the origin, so the difference loses significant
    bits exactly where the interesting signal is. This is cheap here (three columns, once
    per example) and is the same reasoning that keeps the encoder out of bf16.
    """
    p = torch.as_tensor(poses, dtype=torch.float64)
    if p.dim() != 2 or p.shape[1] != POSE_DIM:
        raise ValueError(
            f"poses must be (N, {POSE_DIM}) of (x, y, theta), got {tuple(p.shape)}"
        )
    if p.shape[0] == 0:
        # A legal case, not an error: a window can contain no marker occurrences at all.
        return p.new_zeros((0, POSE_DIM)).float()

    x0, y0, t0 = p[0, 0], p[0, 1], p[0, 2]
    dx = p[:, 0] - x0
    dy = p[:, 1] - y0
    cos0, sin0 = torch.cos(t0), torch.sin(t0)
    # R(-theta0) applied to the translation: express the offset in the start frame.
    rel_x = cos0 * dx + sin0 * dy
    rel_y = -sin0 * dx + cos0 * dy
    rel_t = wrap_to_pi(p[:, 2] - t0)
    return torch.stack([rel_x, rel_y, rel_t], dim=1).float()


def relative_se2_tail(poses: Any, k: int = 1) -> torch.Tensor:
    """`relative_se2(poses)[-k:]`, shaped `(k, 3)` -- the rollout's per-step form.

    The rollout accumulates the raw rows of the value column and re-derives the ones it
    has just added, because every row is expressed against **row 0 of the whole column**
    and that row has to survive the whole episode. Deliberately the whole-sequence
    function and not an incremental shortcut: the shortcut is algebraically equal and
    would be one more thing that can quietly stop being equal.

    `k > 1` exists for PointNav, where one *step* can write more than one marker: a goal
    announcement writes the agent's pose and the goal's, and both are rows of the same
    column. Feeding them through separate calls would work too; feeding them through the
    same accumulator is what makes the k-th marker's row the k-th row, which is the only
    binding the modality mechanism has.
    """
    if k < 1:
        raise ValueError(f"relative_se2_tail needs k >= 1, got {k}")
    rel = relative_se2(poses)
    if rel.shape[0] < k:
        raise ValueError(
            f"relative_se2_tail({k}) needs at least {k} pose(s), got {rel.shape[0]}"
        )
    return rel[-k:]


def relative_se2_last(poses: Any) -> torch.Tensor:
    """`relative_se2(poses)[-1]`, shaped `(1, 3)`. `relative_se2_tail(poses, 1)`."""
    return relative_se2_tail(poses, 1)
