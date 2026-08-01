"""Per-tick differentials for the action autoencoder.

Why differentials and not the raw chunk: the 10 poses in an `action_chunks` row are
all expressed relative to a *single* anchor (the observation pose), so they form a
near-monotone arc -- heavily correlated, close to a low-dimensional manifold, badly
conditioned, and with the exact-zero atom spread across a moving offset rather than
sitting at the origin. In per-tick body-frame differentials the atom is at exactly
zero in every coordinate, which is the only place a flat decoder region can put it.

The SE(2) composition is done by `tick_differentials()` from
`dump/data_diagnostics/analyze_chunk_motion.py` -- imported, not reimplemented, so
the autoencoder and every existing diagnostic agree bit-for-bit on what a "tick" is.
That function takes a `stride` used to de-overlap consecutive chunks when estimating
tick distributions; the autoencoder models a whole chunk, so it wants `stride=10`
(all 10 differentials). The de-overlapped `stride=5` view is used only for the gate,
so the numbers line up with everything already measured.

Cache format (`.npz`, one per split):
    diffs        (n_chunks, 10, 3) float32   [forward, strafe, dtheta] per tick, metres/radians
    episode_ptr  (n_episodes + 1,) int64     diffs[ptr[i]:ptr[i+1]] are episode i's chunks
    episode_id   (n_episodes,)     object    dataset episode ids, for traceability
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DIAGNOSTICS = REPO / "dump" / "data_diagnostics"
if str(DIAGNOSTICS) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS))

from analyze_chunk_motion import tick_differentials  # noqa: E402

DATASET = REPO / "data" / "continuous_sft"
CACHE = REPO / "dump" / "cnf_head" / "autoencoder" / "cache"

CHUNK_LEN = 10
GATE_STRIDE = 5          # the de-overlapped view every existing diagnostic uses
CHANNELS = ("forward", "strafe", "dtheta")

# Band thresholds, kept identical to dump/data_diagnostics/sweep_analysis.py.
EXACT = 1e-4             # 0.1 mm -- below this the robot is stopped, not moving slowly


def build_cache(split: str, max_episodes: int | None = None, seed: int = 0,
                out_dir: Path = CACHE) -> Path:
    """Extract all-10 per-tick differentials for `split` and cache them."""
    from datasets import load_from_disk

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"diffs_{split}.npz"

    ds = load_from_disk(str(DATASET))[split].select_columns(
        ["episode_id", "action_chunks"])
    idx = np.arange(len(ds))
    if max_episodes and len(ds) > max_episodes:
        idx = np.sort(np.random.default_rng(seed).choice(len(ds), max_episodes, False))
    ds.set_format("numpy")

    parts, ptr, ids = [], [0], []
    for k, i in enumerate(idx):
        row = ds[int(i)]
        chunks = row["action_chunks"]
        d = tick_differentials(chunks, stride=CHUNK_LEN).reshape(-1, CHUNK_LEN, 3)
        parts.append(d.astype(np.float32))
        ptr.append(ptr[-1] + len(d))
        ids.append(str(row["episode_id"]))
        if (k + 1) % 2000 == 0:
            print(f"  {split}: {k + 1}/{len(idx)} episodes, "
                  f"{ptr[-1]:,} chunks", flush=True)

    diffs = np.concatenate(parts, axis=0)
    np.savez(path, diffs=diffs, episode_ptr=np.asarray(ptr, dtype=np.int64),
             episode_id=np.asarray(ids, dtype=object))
    print(f"-> {path}  {diffs.shape}  ({diffs.nbytes / 1e6:.0f} MB)")
    return path


def load_cache(split: str, out_dir: Path = CACHE):
    z = np.load(out_dir / f"diffs_{split}.npz", allow_pickle=True)
    return z["diffs"], z["episode_ptr"], z["episode_id"]


def channel_scale(diffs: np.ndarray, q: float = 99.0) -> np.ndarray:
    """Per-channel scale for normalisation: the q-th percentile of |value|.

    Scaling only, never centring. A mean shift would move the data's exact-zero
    atom off the origin, and the decoder's flat region is anchored at the origin by
    construction -- the whole mechanism depends on 0 -> 0.
    """
    a = np.abs(diffs.reshape(-1, 3))
    return np.percentile(a, q, axis=0).astype(np.float32)


def gate_view(diffs: np.ndarray, stride: int = GATE_STRIDE) -> np.ndarray:
    """(n, 10, 3) -> (n * stride, 3), the de-overlapped tick view used for reporting."""
    return diffs[:, :stride, :].reshape(-1, 3)


def flat_view(diffs: np.ndarray) -> np.ndarray:
    """(n, 10, 3) -> (n * 10, 3), every differential (what the AE actually models)."""
    return diffs.reshape(-1, 3)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["validation", "train"])
    p.add_argument("--max-train-episodes", type=int, default=None)
    a = p.parse_args()
    for s in a.splits:
        build_cache(s, max_episodes=a.max_train_episodes if s == "train" else None)
