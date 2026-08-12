"""Tests for the two policy-loss variants configurable via
RLAlgoConfig.policy_loss.name (looked up through verl's
get_policy_loss_fn, which is the source of truth for the loss math itself
-- these tests just pin LongNav's config wiring into it).

`vanilla` is fully deterministic. `clip_cov` uses an unseeded torch.randperm
internally (per the plan's decision #3), so its snapshot needs an explicit
seed -- documented here, not "just works" like the rest of this tier.
"""
import torch

from verl.trainer.ppo.core_algos import get_policy_loss_fn

from _fixtures import make_rl_algo_config
from _snapshot import assert_matches_snapshot


def _synthetic_loss_inputs():
    torch.manual_seed(0)
    old_log_prob = torch.log(torch.rand(3, 5) * 0.5 + 0.25)
    log_prob = old_log_prob + (torch.rand(3, 5) - 0.5) * 0.2
    advantages = torch.randn(3, 5)
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    return old_log_prob, log_prob, advantages, response_mask


def test_vanilla_policy_loss():
    old_log_prob, log_prob, advantages, response_mask = _synthetic_loss_inputs()
    config = make_rl_algo_config(clip_ratio=0.2)

    loss_fn = get_policy_loss_fn("vanilla")
    pg_loss, metrics = loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
    )

    assert_matches_snapshot(
        "policy_loss_vanilla__base",
        scalars={"pg_loss": pg_loss.item(), **{k: (v.item() if torch.is_tensor(v) else v) for k, v in metrics.items()}},
        tensors={},
    )


def test_clip_cov_policy_loss_seeded():
    """clip_cov's internal torch.randperm makes this non-reproducible unless
    seeded immediately before the call -- required for a stable snapshot,
    not optional like the other estimators in this tier."""
    old_log_prob, log_prob, advantages, response_mask = _synthetic_loss_inputs()
    config = make_rl_algo_config(clip_ratio=0.2)
    config.policy_loss.clip_cov_ratio = 0.5  # small batch: need a large enough ratio to clip anything
    config.policy_loss.clip_cov_ub = 5.0
    config.policy_loss.clip_cov_lb = 0.0

    loss_fn = get_policy_loss_fn("clip_cov")

    torch.manual_seed(42)
    pg_loss, metrics = loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
    )

    assert_matches_snapshot(
        "policy_loss_clip_cov__seeded_base",
        scalars={"pg_loss": pg_loss.item(), **{k: (v.item() if torch.is_tensor(v) else v) for k, v in metrics.items()}},
        tensors={},
    )
