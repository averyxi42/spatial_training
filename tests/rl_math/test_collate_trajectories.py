"""Tests for collate_trajectories (utils/rl_core.py), covering discrete-only,
continuous-only, and value-included trajectory lists, plus two known-gap
pins: the old_log_prob derivation-vs-required-key branch, and the two
silent-failure branches (L57-59, L66-69) that the plan calls out as current
(arguably buggy) behavior to lock down, not fix, ahead of the head-
polymorphism refactor.
"""
import numpy as np
import pytest
import torch

from longnav.utils.rl_core import collate_trajectories


def _discrete_traj(length, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "actions": rng.integers(0, 4, size=length).astype(np.int64),
        "rollout_probs": rng.random((length, 4)).astype(np.float32),
        "rewards": rng.random(length).astype(np.float32),
        "dones": np.zeros(length, dtype=bool),
        "old_log_prob": rng.random(length).astype(np.float32),
    }


def _continuous_traj(length, action_dim=2, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "actions_continuous": rng.random((length, action_dim)).astype(np.float32),
        "rewards": rng.random(length).astype(np.float32),
        "dones": np.zeros(length, dtype=bool),
        "old_log_prob": rng.random(length).astype(np.float32),
    }


def test_discrete_only_collation_shapes_and_mask():
    trajs = [_discrete_traj(5, seed=1), _discrete_traj(3, seed=2)]
    batch = collate_trajectories(trajs)

    assert batch["actions"].shape == (2, 5)
    assert batch["actions"].dtype == torch.long
    assert batch["rewards"].dtype == torch.float32
    assert batch["response_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]


def test_continuous_only_collation_uses_actions_continuous_for_length():
    trajs = [_continuous_traj(4, seed=1), _continuous_traj(6, seed=2)]
    batch = collate_trajectories(trajs)

    assert batch["actions_continuous"].shape == (2, 6, 2)
    assert batch["actions_continuous"].dtype == torch.float32
    assert "actions" not in batch
    assert batch["response_mask"].sum(dim=1).tolist() == [4, 6]


def test_value_included_collation_squeezes_singleton_last_dim():
    trajs = [_discrete_traj(4, seed=1), _discrete_traj(4, seed=2)]
    for t in trajs:
        # Per-trajectory value estimates are naturally (T, 1) -- one scalar
        # value per step, but shaped with an explicit trailing singleton dim.
        t["values"] = np.random.default_rng(0).random((4, 1)).astype(np.float32)

    batch = collate_trajectories(trajs)

    assert batch["values"].shape == (2, 4)  # squeezed from (2, 4, 1)


def test_old_log_prob_derived_from_old_logprobs_and_actions():
    length = 4
    rng = np.random.default_rng(0)
    trajs = []
    for seed in (1, 2):
        r = np.random.default_rng(seed)
        trajs.append(
            {
                "actions": r.integers(0, 4, size=length).astype(np.int64),
                "old_logprobs": r.random((length, 4)).astype(np.float32),
                "rewards": r.random(length).astype(np.float32),
                "dones": np.zeros(length, dtype=bool),
            }
        )
    batch = collate_trajectories(trajs)

    # old_log_prob must be the gather of old_logprobs at the taken actions.
    expected = batch["old_logprobs"].gather(2, batch["actions"].unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(batch["old_log_prob"], expected)


def test_missing_old_log_prob_and_old_logprobs_raises_keyerror():
    trajs = [
        {
            "actions": np.array([0, 1, 2], dtype=np.int64),
            "rewards": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "dones": np.zeros(3, dtype=bool),
        }
    ]
    with pytest.raises(KeyError):
        collate_trajectories(trajs)


def test_poisoned_numeric_column_raises_string_metadata_skips():
    """The 2026-08-24 contract (audit F1): a column that fails tensor conversion is
    either pure string metadata -- skipped quietly, as episode_label always was -- or
    a POISONED numeric column, which must raise. The old pinned behavior (silently
    dropping any failing key with a print) is how one None from a navmesh-blind step
    erased distance_to_goal from entire batches and starved the distance-kernel
    estimators without an error."""
    import numpy as np

    # String metadata: skipped quietly, everything else collates.
    trajs = [_discrete_traj(3, seed=1), _discrete_traj(3, seed=2)]
    for t in trajs:
        t["scene_meta"] = ["a_scene"] * 3
    batch = collate_trajectories(trajs)
    assert "scene_meta" not in batch
    assert "actions" in batch

    # A None-poisoned numeric column: loud, naming the key.
    trajs = [_discrete_traj(3, seed=1), _discrete_traj(3, seed=2)]
    for t in trajs:
        t["distance_to_goal"] = np.array([1.0, None, 2.0], dtype=object)
    with pytest.raises(RuntimeError, match="distance_to_goal"):
        collate_trajectories(trajs)

    # A dict-valued column: also loud (not metadata -- a producer bug).
    trajs = [_discrete_traj(3, seed=1), _discrete_traj(3, seed=2)]
    for t in trajs:
        t["uncollatable"] = [{"nested": "dict"}] * 3
    with pytest.raises(RuntimeError, match="uncollatable"):
        collate_trajectories(trajs)


def test_pad_sequence_failure_bleeds_previous_key_known_gap():
    """Pin current (buggy) behavior: when pad_sequence() raises for a given
    key (e.g. mismatched trailing dims across trajectories), the code prints
    a diagnostic but does NOT `continue` or skip storing the key -- `padded`
    still holds whatever the *previous* iterated key produced, and that
    stale tensor gets assigned to the failing key's slot in the batch."""
    trajs = [_discrete_traj(3, seed=1), _discrete_traj(3, seed=2)]
    for i, t in enumerate(trajs):
        # Mismatched trailing dims (3 vs 5) across trajectories -> pad_sequence raises.
        feat_dim = 3 if i == 0 else 5
        t["mismatched_feature"] = np.zeros((3, feat_dim), dtype=np.float32)

    batch = collate_trajectories(trajs)

    # dict iteration order == insertion order: "old_log_prob" is the last key
    # inserted by _discrete_traj before "mismatched_feature", so "padded" at
    # the point of failure still holds the (successfully padded)
    # "old_log_prob" tensor from the previous loop iteration.
    assert "mismatched_feature" in batch
    assert torch.equal(batch["mismatched_feature"], batch["old_log_prob"])
