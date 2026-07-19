"""Tests for apply_hybrid_splitting (utils/rl_core.py, discrete-only per the
regression-test plan's decision #3). Hand-computed expectations against a
small synthetic batch with known failure/success trajectories and integer-
index oracle actions, verifying both the "rescue" (worst-k-of-failures)
trajectory-level selection and the "surgical" FP/FN-stop detection that
applies regardless of trajectory-level selection.
"""
import torch

from longnav.utils.rl_core import apply_hybrid_splitting


def _build_batch():
    # 4 trajectories x 4 steps, all fully valid (no padding) to keep the
    # hand-computed expectation simple.
    actions = torch.tensor(
        [
            [1, 1, 1, 1],  # traj 0: never stops
            [1, 1, 1, 1],  # traj 1: never stops
            [1, 1, 0, 1],  # traj 2: agent stops at t=2 (oracle disagrees -> FP)
            [1, 0, 1, 1],  # traj 3: agent stops at t=1 (oracle disagrees -> FP)
        ],
        dtype=torch.long,
    )
    # Integer-index oracle actions (never stop) -- exercises the non-one-hot path.
    oracle_actions = torch.ones_like(actions)
    response_mask = torch.ones_like(actions)
    rewards = torch.zeros_like(actions, dtype=torch.float32)  # terminal reward 0.0 <= 0.1 for all -> all "failures"
    # min(returns) per trajectory, in increasing order of "badness":
    # traj0=-5 (worst), traj1=-3, traj2=-1, traj3=0 (best-of-the-failures).
    returns = torch.tensor(
        [
            [-5.0, -4.0, -3.0, -2.0],
            [-3.0, -2.0, -1.0, 0.0],
            [-1.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0, 3.0],
        ]
    )
    return {
        "actions": actions,
        "oracle_actions": oracle_actions,
        "response_mask": response_mask,
        "rewards": rewards,
        "returns": returns,
    }


def test_rescue_selects_worst_k_failures_and_surgical_catches_fp_everywhere():
    batch = _build_batch()

    # 4 failure trajectories, dagger_percentile=0.5 -> k=2: the two worst
    # (traj0, traj1) get rescued into full-episode DAgger supervision.
    result = apply_hybrid_splitting(batch, dagger_percentile=0.5, only_failures=True, stop_action_id=0)

    expected_dagger_mask = torch.tensor(
        [
            [1, 1, 1, 1],  # traj 0: rescued (worst return)
            [1, 1, 1, 1],  # traj 1: rescued
            [0, 0, 1, 0],  # traj 2: NOT rescued, but surgical FP at t=2 still caught
            [0, 1, 0, 0],  # traj 3: NOT rescued, but surgical FP at t=1 still caught
        ]
    )
    expected_response_mask = 1 - expected_dagger_mask  # fully valid batch: PPO mask is just the complement

    assert torch.equal(result["dagger_mask"], expected_dagger_mask)
    assert torch.equal(result["response_mask"], expected_response_mask)


def test_mutates_batch_in_place():
    batch = _build_batch()
    result = apply_hybrid_splitting(batch, dagger_percentile=0.5, only_failures=True, stop_action_id=0)
    assert result is batch
    assert "dagger_mask" in batch


def test_zero_failures_disables_rescue_but_keeps_surgical():
    """If terminal rewards are all > 0.1 (no failures), can_use_oracle is
    False everywhere, so n_failures=0 and the rescue path selects nothing --
    only the surgical FP/FN detection can still populate dagger_mask."""
    batch = _build_batch()
    batch["rewards"][:, -1] = 1.0  # every trajectory now "succeeds"

    result = apply_hybrid_splitting(batch, dagger_percentile=0.5, only_failures=True, stop_action_id=0)

    expected_dagger_mask = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],  # surgical FP survives even with rescue disabled
            [0, 1, 0, 0],
        ]
    )
    assert torch.equal(result["dagger_mask"], expected_dagger_mask)
