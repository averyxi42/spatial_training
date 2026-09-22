from types import SimpleNamespace
import pytest
import torch
from test_navigation_credit import future_outcomes
from longnav.utils.train_loop import compute_advantages_and_returns
from longnav.utils.rl_core import compute_navigation_metric_advantages


@pytest.mark.parametrize('mode', ['shared_future', 'shared_stop_baseline', 'separate_stop_baseline'])
@pytest.mark.parametrize('scale', [None, 8.0])
def test_actor_scale_leaves_returns_and_baseline_in_reward_units(mode, scale):
    batch = future_outcomes()
    batch['rewards'] = torch.zeros_like(batch['success'])
    expected = compute_navigation_metric_advantages(batch, mode)
    cfg = SimpleNamespace(training=SimpleNamespace(rl_config=SimpleNamespace(navigation_credit=mode, navigation_advantage_scale=scale)))
    actual, _ = compute_advantages_and_returns(batch, None, cfg)
    factor = 1 if scale is None else (8 if mode.startswith('shared_') else 4) / scale
    torch.testing.assert_close(actual['advantages'], expected[0]*factor)
    torch.testing.assert_close(actual['stop_advantages'], expected[1]*factor)
    torch.testing.assert_close(actual['returns'], expected[2])
    torch.testing.assert_close(actual['stop_returns'], expected[3])
    torch.testing.assert_close(actual['baseline'], expected[4])
