from types import SimpleNamespace

import torch

from longnav.utils.train_loop import compute_advantages_and_returns


def test_bootstrap_only_applies_to_budget_cap():
    captured = {}

    def estimator(*, token_level_rewards, values, response_mask, config):
        captured["rewards"] = token_level_rewards.clone()
        return token_level_rewards, token_level_rewards

    cfg = SimpleNamespace(
        training=SimpleNamespace(
            rl_config=SimpleNamespace(bootstrap_truncated=True, gamma=0.9)
        )
    )
    batch = {
        "rewards": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "values": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        "response_mask": torch.tensor([[True, True], [True, True]]),
        "bootstrap_eligible": torch.tensor([[False, True], [False, False]]),
    }

    compute_advantages_and_returns(batch, estimator, cfg)

    torch.testing.assert_close(
        captured["rewards"], torch.tensor([[1.0, 7.4], [3.0, 4.0]])
    )
    assert torch.equal(batch["rewards"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
