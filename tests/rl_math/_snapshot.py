"""Minimal committed-JSON snapshot helper for the pure-math tier.

Schema (see tests/snapshots/rl_math/*.json), one JSON object per case:
{
  "case": "<name>",
  "scalars": {...},                # plain floats/ints/bools/strings
  "tensors": {"<key>": {"shape": [...], "dtype": "...", "mean": .., "std": .., "sample": [...]}},
  "tolerance": {"atol": .., "rtol": ..}
}

Capture is deliberate: set LONGNAV_UPDATE_SNAPSHOTS=1 to (re)write a
snapshot instead of asserting against it. This is the pure-math tier's
version of Phase 6's `--update-snapshot` flag -- the forward-pass tier
(GPU, real checkpoint) is not implemented yet and will need its own
numeric-tolerance calibration pass before it can use committed snapshots
the same way (see tests/forward/README.md).

Every function here is fully deterministic given fixed inputs -- no model,
no GPU, no Ray -- so unlike the forward-pass tier, no tolerance calibration
step is needed: rtol ~1e-5 catches real regressions without chasing kernel
jitter that doesn't exist on this tier.
"""
import json
import os

import numpy as np
import torch

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "snapshots", "rl_math")
DEFAULT_TOLERANCE = {"atol": 1e-5, "rtol": 1e-5}


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().float().numpy()
    return np.asarray(t, dtype=np.float64)


def summarize_tensor(t, n_sample=8):
    arr = _to_numpy(t)
    flat = arr.reshape(-1)
    sample = flat[: min(n_sample, flat.size)]
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "std": float(np.std(arr)) if arr.size else 0.0,
        "sample": [float(x) for x in sample],
    }


def _snapshot_path(case_name: str) -> str:
    return os.path.join(SNAPSHOT_DIR, f"{case_name}.json")


def assert_matches_snapshot(case_name: str, scalars: dict, tensors: dict, tolerance: dict = None):
    """Compares (scalars, tensors) against the committed snapshot for
    `case_name`. Set LONGNAV_UPDATE_SNAPSHOTS=1 to write/overwrite the
    snapshot instead of asserting -- a human should review the diff before
    committing a regenerated snapshot, same discipline as Phase 6 describes
    for the (not-yet-implemented) forward-pass tier.
    """
    tolerance = tolerance or DEFAULT_TOLERANCE
    tensor_summaries = {k: summarize_tensor(v) for k, v in tensors.items()}
    path = _snapshot_path(case_name)

    if os.environ.get("LONGNAV_UPDATE_SNAPSHOTS") == "1":
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        payload = {
            "case": case_name,
            "scalars": scalars,
            "tensors": tensor_summaries,
            "tolerance": tolerance,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        return

    if not os.path.exists(path):
        raise AssertionError(
            f"No snapshot at {path}. Run with LONGNAV_UPDATE_SNAPSHOTS=1 to capture it, "
            "then review the values before committing."
        )
    with open(path) as f:
        expected = json.load(f)

    atol, rtol = tolerance["atol"], tolerance["rtol"]

    # Symmetric key-set check first: iterating only the current run's keys
    # would catch an *added* key (missing from the snapshot -> KeyError below)
    # but silently miss a *removed* one (the snapshot's extra key is just
    # never visited). A refactor that drops an output must fail loudly here,
    # not pass because nothing was left around to notice its absence.
    assert set(scalars) == set(expected["scalars"]), (
        f"{case_name}: scalar key set changed. "
        f"missing={set(expected['scalars']) - set(scalars)} "
        f"added={set(scalars) - set(expected['scalars'])}"
    )
    assert set(tensor_summaries) == set(expected["tensors"]), (
        f"{case_name}: tensor key set changed. "
        f"missing={set(expected['tensors']) - set(tensor_summaries)} "
        f"added={set(tensor_summaries) - set(expected['tensors'])}"
    )

    for key, value in scalars.items():
        exp = expected["scalars"].get(key)
        assert exp is not None, f"{case_name}: snapshot missing scalar {key!r}"
        if isinstance(value, (int, float)):
            assert np.isclose(value, exp, atol=atol, rtol=rtol), f"{case_name}: scalar {key!r} mismatch: {value} vs {exp}"
        else:
            assert value == exp, f"{case_name}: scalar {key!r} mismatch: {value!r} vs {exp!r}"

    for key, summary in tensor_summaries.items():
        exp = expected["tensors"].get(key)
        assert exp is not None, f"{case_name}: snapshot missing tensor {key!r}"
        assert summary["shape"] == exp["shape"], f"{case_name}: tensor {key!r} shape mismatch: {summary['shape']} vs {exp['shape']}"
        for stat in ("mean", "std"):
            assert np.isclose(summary[stat], exp[stat], atol=atol, rtol=rtol), (
                f"{case_name}: tensor {key!r} {stat} mismatch: {summary[stat]} vs {exp[stat]}"
            )
        np.testing.assert_allclose(
            summary["sample"], exp["sample"], atol=atol, rtol=rtol, err_msg=f"{case_name}: tensor {key!r} sample mismatch"
        )
