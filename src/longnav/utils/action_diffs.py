"""Per-tick body-frame action differentials — the space every head models.

This is the single source of truth for converting an action chunk into the
representation the policy heads actually predict. It was previously duplicated in
three untracked scripts under `dump/`; anything needing differentials should import
it from here.

## What a chunk is, and why differentials

A dataset row's `action_chunks` is `(n_obs, chunk_len, 3)`: for each observation,
`chunk_len` future poses `(x, y, theta)`, **all expressed relative to a single
anchor** — the pose at that observation. They are cumulative, so `chunk[9]` is
where the robot will be nine ticks later, measured from where it was when it looked.

That is the wrong space to model in. The coordinates are strongly correlated (a
near-monotone arc), their scale grows along the chunk, and "the robot did not move
during tick 7" is not any particular value — it is `chunk[7] == chunk[6]`.

Differentials fix all three: each tick becomes the motion *during* that tick, in the
robot's own frame at the start of it. "Did not move" becomes the origin, in every
coordinate, at every position in the chunk.

## The composition is not subtraction

Consecutive poses are composed in SE(2), not subtracted. Subtracting would express
tick `t`'s motion in the *anchor's* frame; what is wanted is its own frame, so the
world-frame delta is rotated by `-theta_t` first. Getting this wrong yields
differentials that look plausible and silently smear the rotation axis.

The anchor is prepended as the origin so the first differential
(anchor -> `chunk[0]`) is included — a chunk of length L yields L differentials.

## `stride`

Consecutive observations overlap: with `obs_stride_frames=5` and `chunk_len=10`,
half of each chunk repeats the next one. For **distribution estimation** pass
`stride=obs_stride_frames` to de-overlap and avoid double-counting. For **modelling
a chunk** pass `stride=chunk_len` (the default) — every tick of the chunk is a
target.
"""

from __future__ import annotations

import numpy as np

CHUNK_LEN = 10


def wrap(a):
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def tick_differentials(chunks, stride: int = CHUNK_LEN) -> np.ndarray:
    """`(n_obs, chunk_len, 3)` anchor-relative chunks -> `(n_obs * stride, 3)` per-tick
    body-frame `[dx, dy, dtheta]`.

    See the module docstring for why this is a composition rather than a difference,
    and how to choose `stride`.
    """
    chunks = np.asarray(chunks, dtype=np.float64)
    n = len(chunks)
    zero = np.zeros((n, 1, 3), dtype=np.float64)
    poses = np.concatenate([zero, chunks], axis=1)     # prepend the anchor
    a, b = poses[:, :-1, :], poses[:, 1:, :]
    th = a[..., 2]
    c, s = np.cos(th), np.sin(th)
    ddx, ddy = b[..., 0] - a[..., 0], b[..., 1] - a[..., 1]
    out = np.stack([c * ddx + s * ddy,                 # world delta -> body frame
                    -s * ddx + c * ddy,
                    wrap(b[..., 2] - a[..., 2])], axis=-1)
    return out[:, :stride, :].reshape(-1, 3)


def load_diffs_from_dataset(dataset: str, split: str, max_episodes: int | None = None,
                            seed: int = 0, stride: int = CHUNK_LEN) -> np.ndarray:
    """Differentials straight from an HF dataset's `action_chunks` column.

    Any corpus can be read this way; no prebuilt cache is required.
    """
    from datasets import load_from_disk

    ds = load_from_disk(dataset)[split].select_columns(["action_chunks"])
    idx = np.arange(len(ds))
    if max_episodes and len(ds) > max_episodes:
        idx = np.sort(np.random.default_rng(seed).choice(len(ds), max_episodes, False))
    ds.set_format("numpy")
    parts = [tick_differentials(ds[int(i)]["action_chunks"], stride=stride) for i in idx]
    return np.concatenate(parts) if parts else np.zeros((0, 3))
