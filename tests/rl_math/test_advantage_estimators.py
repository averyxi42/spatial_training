"""Pure-math regression tests for the 5 LongNav-specific advantage
estimators in utils/rl_core.py. CPU-only, deterministic given fixed inputs
(no RNG in any of these functions) -- CI-eligible.

Note: the config-default "reinforce_plus_plus" estimator is a verl builtin,
not defined in this repo, so it isn't snapshotted here -- only the 5
kernel/regression variants LongNav adds on top are in scope.
"""
from longnav.utils.rl_core import (
    compute_reinforce_plus_plus_time_kernel_advantage,
    compute_reinforce_plus_plus_linear_time_aware_advantage,
    compute_reinforce_plus_plus_geometric_time_aware_advantage,
    compute_reinforce_plus_plus_distance_kernel_advantage,
    compute_reinforce_plus_plus_distance_kernel_var_norm_advantage,
)

from _fixtures import make_reward_batch, make_distances_for, make_rl_algo_config
from _snapshot import assert_matches_snapshot


def test_time_kernel_advantage():
    rewards, response_mask = make_reward_batch()
    config = make_rl_algo_config(gamma=0.95, time_kernel_sigma=2.0)

    advantages, returns, baseline = compute_reinforce_plus_plus_time_kernel_advantage(
        token_level_rewards=rewards, response_mask=response_mask, config=config
    )

    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert_matches_snapshot(
        "reinforce_plus_plus_time_kernel__base",
        scalars={},
        tensors={"advantages": advantages, "returns": returns, "baseline": baseline},
    )


def test_linear_time_aware_advantage():
    rewards, response_mask = make_reward_batch()
    config = make_rl_algo_config(gamma=0.95)

    advantages, returns = compute_reinforce_plus_plus_linear_time_aware_advantage(
        token_level_rewards=rewards, response_mask=response_mask, config=config
    )

    assert advantages.shape == rewards.shape
    assert_matches_snapshot(
        "reinforce_plus_plus_linear_time_aware__base",
        scalars={},
        tensors={"advantages": advantages, "returns": returns},
    )


def test_geometric_time_aware_advantage():
    rewards, response_mask = make_reward_batch()
    config = make_rl_algo_config(gamma=0.95)

    advantages, returns = compute_reinforce_plus_plus_geometric_time_aware_advantage(
        token_level_rewards=rewards, response_mask=response_mask, config=config
    )

    assert advantages.shape == rewards.shape
    assert_matches_snapshot(
        "reinforce_plus_plus_geometric_time_aware__base",
        scalars={},
        tensors={"advantages": advantages, "returns": returns},
    )


def test_distance_kernel_advantage():
    rewards, response_mask = make_reward_batch()
    distances = make_distances_for(response_mask)
    config = make_rl_algo_config(gamma=0.95, distance_kernel_sigma=0.5, distance_clip_max=None, distance_clip_percentile=None)

    advantages, returns, baseline_table = compute_reinforce_plus_plus_distance_kernel_advantage(
        token_level_rewards=rewards, response_mask=response_mask, config=config, distances=distances
    )

    assert advantages.shape == rewards.shape
    assert_matches_snapshot(
        "reinforce_plus_plus_distance_kernel__base",
        scalars={},
        tensors={"advantages": advantages, "returns": returns},
    )


def test_distance_kernel_var_norm_advantage():
    rewards, response_mask = make_reward_batch()
    distances = make_distances_for(response_mask)
    config = make_rl_algo_config(gamma=0.95, distance_kernel_sigma=0.5, distance_clip_max=None, distance_clip_percentile=None)

    advantages, returns, mean_table, std_table = compute_reinforce_plus_plus_distance_kernel_var_norm_advantage(
        token_level_rewards=rewards, response_mask=response_mask, config=config, distances=distances
    )

    assert advantages.shape == rewards.shape
    assert_matches_snapshot(
        "reinforce_plus_plus_distance_kernel_var_norm__base",
        scalars={},
        tensors={"advantages": advantages, "returns": returns},
    )
