"""Exact/tolerant comparison for stored TensorDict fixtures. No summarizing
(no mean/std reduction) -- every element of every tensor is compared;
integer/bool tensors require exact equality, floating-point tensors use a
tolerance (real GPU forward-pass jitter, see each test file's calibration
note)."""
import torch
from tensordict import TensorDict


def diff_traj_batches(actual: TensorDict, expected: TensorDict, atol: float, rtol: float, exclude_keys=()):
    """`exclude_keys` should be used sparingly and only ever for a
    documented, discovered reason -- see each test file's own comment for
    why a given key is excluded (e.g. production's own
    `np.random.seed(os.getpid())` in RolloutWorker.__init__ makes sampled
    discrete actions inherently non-reproducible across separate process
    launches, which is a real fact about the shipped code, not a test
    convenience)."""
    mismatches = []
    actual_keys = set(actual.keys()) - set(exclude_keys)
    expected_keys = set(expected.keys()) - set(exclude_keys)
    if actual_keys != expected_keys:
        mismatches.append(
            f"key set differs: missing={expected_keys - actual_keys} added={actual_keys - expected_keys}"
        )
        return mismatches

    for k in sorted(actual_keys):
        a, e = actual[k], expected[k]
        if tuple(a.shape) != tuple(e.shape):
            mismatches.append(f"{k}: shape {tuple(a.shape)} vs {tuple(e.shape)}")
            continue
        if a.is_floating_point() or e.is_floating_point():
            if not torch.allclose(a.float(), e.float(), atol=atol, rtol=rtol):
                diff = (a.float() - e.float()).abs()
                mismatches.append(
                    f"{k}: max abs diff {diff.max().item():.6g} (atol={atol}, rtol={rtol}, dtype={a.dtype} vs {e.dtype})"
                )
        else:
            if not torch.equal(a, e):
                n_diff = (a != e).sum().item()
                mismatches.append(f"{k}: {n_diff}/{a.numel()} elements differ (exact equality required, dtype={a.dtype})")
    return mismatches
